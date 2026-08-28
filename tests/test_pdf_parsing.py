from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from taskforge.knowledge import tokenise
from taskforge.pdf_parsing.contracts import DocumentBlock, ParsedDocument
from taskforge.pdf_parsing.hierarchy import (
    build_boundary_aware_flat_units,
    build_flat_units,
    build_parent_child_units,
    build_sliding_window_units,
)
from taskforge.pdf_parsing.mineru_client import MinerUClient, MinerUError
from taskforge.pdf_parsing.mineru_normalizer import normalize_mineru_response
from taskforge.pdf_parsing.native_parser import NativePDFParser
from taskforge.pdf_parsing.quality_gate import evaluate_parse_quality
from taskforge.pdf_parsing.router import PDFParserRouter
from taskforge.pdf_parsing.structure_policy import build_structure_aware_units
from taskforge.pdf_parsing.visual_evidence import (
    OpenAICompatibleVisualEvidenceExtractor,
    enrich_visual_evidence,
)


def _pdf(path: Path, text: str = "Native machine-readable research evidence.") -> None:
    document = canvas.Canvas(str(path))
    document.drawString(72, 720, text)
    document.save()


def _pdf_with_image(path: Path) -> None:
    image_path = path.with_suffix(".png")
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
    document = canvas.Canvas(str(path))
    document.drawString(72, 720, "Figure 1 shows the visual result.")
    document.drawImage(ImageReader(str(image_path)), 72, 620, width=64, height=64)
    document.save()


def _block(
    block_id: str,
    *,
    block_type: str = "paragraph",
    text: str = "evidence",
    page: int = 1,
    structured_content: dict[str, object] | None = None,
) -> DocumentBlock:
    import hashlib

    return DocumentBlock(
        block_id=block_id,
        document_id="pdf:" + "a" * 24,
        parser="native",
        parser_version="test",
        page=page,
        bbox=(0.0, 0.0, 1.0, 1.0),
        reading_order=0,
        block_type=block_type,  # type: ignore[arg-type]
        text=text,
        structured_content=structured_content or {},
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def _parsed(blocks: tuple[DocumentBlock, ...]) -> ParsedDocument:
    return ParsedDocument(
        document_id="pdf:" + "a" * 24,
        source_uri="paper://structure-policy",
        sha256="a" * 64,
        bytes_read=100,
        page_count=max(block.page for block in blocks),
        parser="native",
        parser_version="test",
        parser_backend="native",
        blocks=blocks,
        quality=evaluate_parse_quality(
            blocks,
            page_count=max(block.page for block in blocks),
            parser="native",
        ),
    )


def test_structure_policy_uses_fixed_fallback_without_usable_structure() -> None:
    document = _parsed(
        (
            _block("p1", text="plain paragraph one"),
            _block("p2", text="plain paragraph two"),
        )
    )

    result = build_structure_aware_units(document)

    assert result.policy.name == "flat_fallback_v1"
    assert result.profile.usable_hierarchy is False
    assert {unit.role for unit in result.units} == {"child"}
    assert all(unit.parent_id == unit.unit_id for unit in result.units)


def test_structure_policy_preserves_titles_lists_and_tables_hierarchically() -> None:
    title = _block("title", block_type="title", text="Results").model_copy(
        update={"heading_level": 1}
    )
    document = _parsed(
        (
            title,
            _block("list", block_type="list", text="1. first\n2. second"),
            _block(
                "table",
                block_type="table",
                text="Model | Recall\nA | 0.91",
            ),
        )
    )

    result = build_structure_aware_units(document)

    assert result.policy.name == "structured_parent_child_v1"
    assert result.profile.title_blocks == 1
    assert result.profile.structured_blocks == 2
    parents = [unit for unit in result.units if unit.role == "parent"]
    children = [unit for unit in result.units if unit.role == "child"]
    assert parents and children
    table_children = [unit for unit in children if "table" in unit.block_types]
    assert len(table_children) == 1
    assert table_children[0].block_ids == ("table",)


def test_hybrid_policy_splits_oversized_prose_without_changing_current_chunker() -> None:
    title = _block("title-long", block_type="title", text="Methods").model_copy(
        update={"heading_level": 1}
    )
    document = _parsed(
        (title, _block("long", text="evidence " * 700))
    )

    current_children = [
        unit
        for unit in build_parent_child_units(document)
        if unit.role == "child"
    ]
    optimized_children = [
        unit
        for unit in build_structure_aware_units(document).units
        if unit.role == "child"
    ]

    assert len(current_children) == 2
    assert max(len(tokenise(unit.text)) for unit in current_children) > 500
    assert len(optimized_children) > len(current_children)
    assert max(len(tokenise(unit.text)) for unit in optimized_children) <= 500


def test_boundary_aware_flat_keeps_natural_blocks_near_target() -> None:
    document = _parsed(
        (
            _block("p1", text="First paragraph. " + "alpha " * 90),
            _block("p2", text="Second paragraph. " + "beta " * 90),
            _block("p3", text="Third paragraph. " + "gamma " * 90),
        )
    )

    units = build_boundary_aware_flat_units(
        document,
        target_chars=1_000,
        min_chars=500,
        max_chars=1_300,
        search_chars=200,
    )

    assert len(units) == 2
    assert units[0].block_ids == ("p1", "p2")
    assert units[1].block_ids == ("p3",)
    assert all(len(unit.text) <= 1_300 for unit in units)


def test_boundary_aware_flat_does_not_cross_heading_sections() -> None:
    methods = _block("methods", block_type="title", text="2 Methods").model_copy(
        update={"heading_level": 1}
    )
    results = _block("results", block_type="title", text="3 Results").model_copy(
        update={"heading_level": 1}
    )
    document = _parsed(
        (
            methods,
            _block("m1", text="Method detail. " + "method " * 120),
            results,
            _block("r1", text="Result detail. " + "result " * 120),
        )
    )

    units = build_boundary_aware_flat_units(
        document,
        target_chars=1_000,
        min_chars=256,
        max_chars=1_500,
        search_chars=300,
    )

    assert len(units) == 2
    assert units[0].block_ids == ("methods", "m1")
    assert units[1].block_ids == ("results", "r1")
    assert "3 Results" not in units[0].text


def test_boundary_aware_flat_keeps_tables_atomic_and_attaches_caption() -> None:
    document = _parsed(
        (
            _block("intro", text="The experiment is summarized below. " + "text " * 60),
            _block("caption", block_type="caption", text="Table 1. Results"),
            _block(
                "table",
                block_type="table",
                text="Model | Recall\nA | 0.91\nB | 0.88",
            ),
            _block("after", text="The results confirm the hypothesis. " + "text " * 60),
        )
    )

    units = build_boundary_aware_flat_units(
        document,
        target_chars=700,
        min_chars=256,
        max_chars=1_000,
        search_chars=200,
    )

    table_units = [unit for unit in units if "table" in unit.block_types]
    assert len(table_units) == 1
    assert table_units[0].block_ids.index("caption") + 1 == table_units[0].block_ids.index("table")
    assert "Table 1. Results" in table_units[0].text


def test_boundary_aware_flat_splits_oversized_prose_without_losing_text() -> None:
    original = " ".join(f"Sentence {index}." for index in range(500))
    document = _parsed((_block("long", text=original),))

    units = build_boundary_aware_flat_units(
        document,
        target_chars=2_000,
        min_chars=1_000,
        max_chars=2_600,
        search_chars=400,
    )

    normalized_original = " ".join(original.split())
    normalized_chunks = " ".join(" ".join(unit.text.split()) for unit in units)
    assert len(units) >= 3
    assert normalized_chunks == normalized_original
    assert all(len(unit.text) <= 2_600 for unit in units)


@pytest.mark.asyncio
async def test_native_parser_extracts_original_pdf_text(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    _pdf(path)

    parsed = await NativePDFParser().parse(path, source_uri="paper://one")

    assert parsed.parser == "native"
    assert parsed.quality.status == "ready"
    assert parsed.quality.ocr_used is False
    assert "machine-readable" in " ".join(block.text for block in parsed.blocks)


@pytest.mark.asyncio
async def test_native_parser_flags_embedded_images_for_structured_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper.pdf"
    _pdf_with_image(path)

    parsed = await NativePDFParser().parse(path, source_uri="paper://visual")

    assert parsed.quality.status == "visual_pending"
    assert any(block.block_type == "image" for block in parsed.blocks)


def test_quality_gate_marks_unrendered_visual_content_pending() -> None:
    blocks = [
        _block("text", text="The paper reports Figure 2."),
        _block(
            "figure",
            block_type="image",
            text="",
            structured_content={"image_path": "figure-2.png"},
        ),
    ]

    quality = evaluate_parse_quality(blocks, page_count=1, parser="mineru")

    assert quality.status == "visual_pending"
    assert quality.visual_unparsed_count == 1


def test_quality_gate_rejects_empty_table_metadata_as_content() -> None:
    blocks = [
        _block("text", text="The paper reports Table 1."),
        _block(
            "table",
            block_type="table",
            text="",
            structured_content={"coordinate_space": "mineru_1000", "type": "table"},
        ),
    ]

    quality = evaluate_parse_quality(blocks, page_count=1, parser="mineru")

    assert quality.status == "table_failed"
    assert quality.empty_table_count == 1


@pytest.mark.asyncio
async def test_router_keeps_recovered_text_when_one_table_failed() -> None:
    text = _block("body", text="The rest of the paper remains searchable.")
    empty_table = _block(
        "table",
        block_type="table",
        text="",
        structured_content={"coordinate_space": "mineru_1000", "type": "table"},
    )
    quality = evaluate_parse_quality(
        (text, empty_table),
        page_count=1,
        parser="mineru",
    )

    class StubParser:
        name = "mineru"

        async def parse(self, path: Path, *, source_uri: str) -> ParsedDocument:
            return ParsedDocument(
                document_id=text.document_id,
                source_uri=source_uri,
                sha256="a" * 64,
                bytes_read=100,
                page_count=1,
                parser="mineru",
                parser_version="test",
                parser_backend="pipeline",
                blocks=(text, empty_table),
                quality=quality,
            )

    parser = StubParser()
    parsed = await PDFParserRouter(
        parser,
        parser,
        backend="mineru",
    ).parse(Path("ignored.pdf"), source_uri="paper://partial-table")

    assert parsed.quality.status == "table_failed"
    assert parsed.quality.empty_table_count == 1
    assert any(block.text == "The rest of the paper remains searchable." for block in parsed.blocks)


def test_quality_gate_ignores_headers_already_excluded_from_index() -> None:
    body = _block("body", text="indexable research evidence")
    headers = tuple(
        _block(
            f"header-{page}",
            block_type="header",
            text="A preprint - January 25, 2019",
            page=page,
        ).model_copy(
            update={
                "indexable": False,
                "bbox": (600.0, 42.0, 883.0, 58.0),
                "structured_content": {"coordinate_space": "mineru_1000"},
            }
        )
        for page in range(1, 6)
    )

    quality = evaluate_parse_quality(
        (body, *headers),
        page_count=5,
        parser="mineru",
    )

    assert quality.repeated_header_ratio == 0.0


def test_mineru_v3_content_list_is_normalized_without_schema_leakage() -> None:
    raw = {
        "results": {
            "paper.pdf": {
                "parse_method": "ocr",
                "content_list_v2": [
                    {
                        "page_idx": 0,
                        "items": [
                            {
                                "type": "title",
                                "bbox": [50, 40, 950, 120],
                                "content": {
                                    "title_content": [
                                        {"type": "text", "content": "Method"}
                                    ],
                                    "level": 1,
                                },
                            },
                            {
                                "type": "table",
                                "bbox": [50, 200, 950, 700],
                                "content": {
                                    "table_body": "| Model | Recall |\n| A | 0.91 |"
                                },
                            },
                        ],
                    }
                ],
            }
        }
    }

    parsed = normalize_mineru_response(
        raw,
        filename="paper.pdf",
        source_uri="paper://one",
        sha256="b" * 64,
        bytes_read=100,
        page_count=1,
        parser_version="3.0.0",
        parser_backend="hybrid-engine",
    )

    assert [block.block_type for block in parsed.blocks] == ["title", "table"]
    assert parsed.blocks[0].text == "Method"
    assert "0.91" in parsed.blocks[1].text
    assert parsed.blocks[1].structured_content["coordinate_space"] == "mineru_1000"
    assert parsed.quality.ocr_used is True


def test_mineru_flat_text_level_is_normalized_as_a_title() -> None:
    raw = {
        "results": {
            "paper.pdf": {
                "content_list": [
                    {
                        "type": "text",
                        "text": "2.1. Methods",
                        "text_level": 2,
                        "bbox": [50, 40, 950, 120],
                        "page_idx": 0,
                    },
                    {
                        "type": "text",
                        "text": "We evaluate the proposed method.",
                        "bbox": [50, 140, 950, 220],
                        "page_idx": 0,
                    },
                ]
            }
        }
    }

    parsed = normalize_mineru_response(
        raw,
        filename="paper.pdf",
        source_uri="paper://flat-heading",
        sha256="d" * 64,
        bytes_read=100,
        page_count=1,
        parser_version="3.4.4",
        parser_backend="pipeline",
    )

    assert [block.block_type for block in parsed.blocks] == ["title", "paragraph"]
    assert parsed.blocks[0].heading_level == 3
    assert parsed.blocks[1].heading_level is None


def test_mineru_list_items_become_individual_lineage_preserving_blocks() -> None:
    raw = {
        "results": {
            "paper.pdf": {
                "content_list": [
                    {
                        "type": "list",
                        "sub_type": "ref_text",
                        "list_items": [
                            "[1] First reference.",
                            "[2] Second reference.",
                            "[3] Third reference.",
                        ],
                        "bbox": [50, 100, 950, 800],
                        "page_idx": 0,
                    }
                ]
            }
        }
    }

    parsed = normalize_mineru_response(
        raw,
        filename="paper.pdf",
        source_uri="paper://list",
        sha256="e" * 64,
        bytes_read=100,
        page_count=1,
        parser_version="3.4.4",
        parser_backend="pipeline",
    )

    assert [block.text for block in parsed.blocks] == [
        "[1] First reference.",
        "[2] Second reference.",
        "[3] Third reference.",
    ]
    origins = {
        block.structured_content["mineru_list_group_hash"]
        for block in parsed.blocks
    }
    assert len(origins) == 1
    assert [
        block.structured_content["mineru_list_item_index"]
        for block in parsed.blocks
    ] == [0, 1, 2]


def test_dense_visual_labels_do_not_create_false_section_parents() -> None:
    labels = [
        {
            "type": "text",
            "text": f"Community label {index}",
            "text_level": 2,
            "bbox": [50 + index, 100, 200 + index, 120],
            "page_idx": 0,
        }
        for index in range(8)
    ]
    raw = {
        "results": {
            "paper.pdf": {
                "content_list": [
                    *labels,
                    {
                        "type": "image",
                        "img_path": "images/diagram.png",
                        "bbox": [40, 80, 960, 800],
                        "page_idx": 0,
                    },
                ]
            }
        }
    }

    parsed = normalize_mineru_response(
        raw,
        filename="paper.pdf",
        source_uri="paper://diagram-labels",
        sha256="f" * 64,
        bytes_read=100,
        page_count=1,
        parser_version="3.4.4",
        parser_backend="pipeline",
    )

    labels = [block for block in parsed.blocks if block.text.startswith("Community")]
    assert all(block.block_type == "paragraph" for block in labels)
    assert all(block.heading_level is None for block in labels)
    assert all(
        block.structured_content["heading_demoted_reason"] == "dense_visual_labels"
        for block in labels
    )
    assert not [block for block in parsed.blocks if block.block_type == "title"]


def test_mineru_legacy_visual_content_keeps_caption_analysis_and_locator() -> None:
    raw = {
        "results": {
            "paper.pdf": {
                "content_list": [
                    {
                        "type": "chart",
                        "page_idx": 0,
                        "bbox": [0.1, 0.2, 0.8, 0.7],
                        "img_path": "images/chart.jpg",
                        "chart_caption": ["Figure 2. Recall by model."],
                        "content": "| Model | Recall |\n| A | 0.91 |",
                    }
                ]
            }
        }
    }

    parsed = normalize_mineru_response(
        raw,
        filename="paper.pdf",
        source_uri="paper://visual",
        sha256="c" * 64,
        bytes_read=100,
        page_count=1,
        parser_version="3.4.0",
        parser_backend="hybrid-engine",
    )

    block = parsed.blocks[0]
    assert block.block_type == "chart"
    assert "Figure 2" in block.text and "0.91" in block.text
    assert block.structured_content["coordinate_space"] == "normalized"
    assert block.image_artifact_id == f"mineru:{'c' * 16}:images/chart.jpg"
    assert parsed.quality.status == "ready"


@pytest.mark.asyncio
async def test_mineru_client_pins_health_and_caches_raw_response(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    _pdf(path)
    calls = {"health": 0, "parse": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            calls["health"] += 1
            return httpx.Response(200, json={"version": "3.0.1"})
        if request.url.path == "/file_parse":
            calls["parse"] += 1
            return httpx.Response(
                200,
                json={
                    "results": {
                        "paper.pdf": {
                            "content_list_v2": [
                                {
                                    "page_idx": 0,
                                    "items": [
                                        {
                                            "type": "paragraph",
                                            "bbox": [10, 10, 900, 100],
                                            "content": {
                                                "paragraph_content": "cached evidence"
                                            },
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                },
            )
        return httpx.Response(404)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    parser = MinerUClient(
        "http://127.0.0.1:8000",
        tmp_path / "cache",
        expected_version="3.0.1",
        client=http,
    )

    first = await parser.parse(path, source_uri="paper://one")
    second = await parser.parse(path, source_uri="paper://one")

    assert first.parser_version == "3.0.1"
    assert second.blocks[0].text == "cached evidence"
    assert calls == {"health": 1, "parse": 1}
    assert len(list((tmp_path / "cache").glob("*.mineru.json"))) == 1
    await http.aclose()


@pytest.mark.asyncio
async def test_mineru_auto_explicitly_requests_ocr_for_image_only_pdf(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scan.png"
    Image.new("RGB", (600, 800), color=(255, 255, 255)).save(image_path)
    path = tmp_path / "scan.pdf"
    document = canvas.Canvas(str(path))
    document.drawImage(ImageReader(str(image_path)), 0, 0, width=600, height=800)
    document.save()
    request_body = b""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        if request.url.path == "/health":
            return httpx.Response(200, json={"version": "3.4.4"})
        request_body = await request.aread()
        return httpx.Response(
            200,
            json={
                "results": {
                    "scan.pdf": {
                        "content_list": [
                            {
                                "type": "text",
                                "page_idx": 0,
                                "text": "OCR recovered research evidence.",
                            }
                        ]
                    }
                }
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    parser = MinerUClient(
        "http://127.0.0.1:8000",
        tmp_path / "cache",
        expected_version="3.4.4",
        parse_method="auto",
        client=http,
    )

    parsed = await parser.parse(path, source_uri="paper://scan")
    wrapper = json.loads(next((tmp_path / "cache").glob("*.mineru.json")).read_text())

    assert b'name="parse_method"\r\n\r\nocr' in request_body
    assert wrapper["parse_method"] == "ocr"
    assert parsed.quality.ocr_used is True
    await http.aclose()


@pytest.mark.asyncio
async def test_mineru_client_materializes_returned_visual_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    _pdf(path)
    image_bytes = b"visual-artifact"

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"version": "3.4.0"})
        return httpx.Response(
            200,
            json={
                "results": {
                    "paper.pdf": {
                        "content_list": [
                            {
                                "type": "image",
                                "page_idx": 0,
                                "bbox": [10, 10, 900, 900],
                                "img_path": "images/figure.png",
                                "image_caption": ["Figure 1. Architecture."],
                            }
                        ],
                        # MinerU 3.4.x uses a basename here while img_path in
                        # content_list is prefixed with "images/".
                        "images": {"figure.png": base64.b64encode(image_bytes).decode()},
                    }
                }
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    parser = MinerUClient(
        "http://127.0.0.1:8000",
        tmp_path / "cache",
        expected_version="3.4.0",
        client=http,
    )

    parsed = await parser.parse(path, source_uri="paper://visual")
    artifact = Path(parsed.blocks[0].image_artifact_id or "")

    assert artifact.is_file()
    assert artifact.read_bytes() == image_bytes
    await http.aclose()


@pytest.mark.asyncio
async def test_mineru_client_requests_image_analysis(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    _pdf(path)
    request_body = b""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        if request.url.path == "/health":
            return httpx.Response(200, json={"version": "3.4.4"})
        request_body = await request.aread()
        return httpx.Response(
            200,
            json={
                "results": {
                    "paper.pdf": {
                        "content_list": [
                            {"type": "text", "page_idx": 0, "text": "evidence"}
                        ]
                    }
                }
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    parser = MinerUClient(
        "http://127.0.0.1:8000",
        tmp_path / "cache",
        expected_version="3.4.4",
        client=http,
    )

    await parser.parse(path, source_uri="paper://visual-analysis")

    assert b'name="image_analysis"' in request_body
    assert b"true" in request_body
    await http.aclose()


@pytest.mark.asyncio
async def test_mineru_client_rejects_unpinned_runtime_version(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"version": "3.1.0"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    parser = MinerUClient(
        "http://127.0.0.1:8000",
        tmp_path / "cache",
        expected_version="3.0.1",
        client=http,
    )

    with pytest.raises(MinerUError, match="version mismatch"):
        await parser.health()
    await http.aclose()


@pytest.mark.asyncio
async def test_visual_extractor_caches_text_model_safe_evidence(tmp_path: Path) -> None:
    artifact_root = tmp_path / "mineru-cache"
    image_path = artifact_root / "images" / "figure.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = request.read().decode()
        assert "data:image/png;base64," in body
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"visual_type":"diagram","caption":"Figure 1",'
                                '"axes":null,"legends":[],"data_points":[],'
                                '"nodes":[{"label":"Encoder"}],"edges":[],'
                                '"textual_rendering":"Figure 1 contains an Encoder node.",'
                                '"confidence":0.9,"warnings":[]}'
                            )
                        }
                    }
                ]
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    extractor = OpenAICompatibleVisualEvidenceExtractor(
        api_key="test-key",
        model="vision-model-v1",
        base_url="http://127.0.0.1:9000/v1",
        cache_root=tmp_path / "visual-cache",
        artifact_root=artifact_root,
        client=http,
    )
    block = _block(
        "figure",
        block_type="image",
        text="Figure 1.",
        structured_content={"visual_analysis_status": "pending"},
    ).model_copy(update={"image_artifact_id": str(image_path.resolve())})

    first = await extractor.extract(block)
    second = await extractor.extract(block)

    assert first.textual_rendering == second.textual_rendering
    assert first.image_sha256 == second.image_sha256
    assert calls == 1
    await http.aclose()


@pytest.mark.asyncio
async def test_visual_enrichment_preserves_provenance_and_clears_pending() -> None:
    class StubExtractor:
        async def extract(self, block: DocumentBlock):  # type: ignore[no-untyped-def]
            from taskforge.pdf_parsing.contracts import VisualEvidence

            return VisualEvidence(
                visual_id=f"visual:{block.block_id}",
                page=block.page,
                bbox=block.bbox,
                visual_type="chart",
                caption="Recall by model",
                textual_rendering="Model A has recall 0.91.",
                confidence=0.95,
                image_artifact_id=str(block.image_artifact_id),
                extractor="stub-vlm",
                extractor_version="stub-v1",
                image_sha256="f" * 64,
            )

    figure = _block(
        "figure",
        block_type="chart",
        text="Recall by model",
        structured_content={"visual_analysis_status": "pending"},
    ).model_copy(update={"image_artifact_id": "C:/parser-cache/figure.png"})
    document = ParsedDocument(
        document_id="pdf:" + "a" * 24,
        source_uri="paper://visual",
        sha256="a" * 64,
        bytes_read=100,
        page_count=1,
        parser="mineru",
        parser_version="test",
        parser_backend="hybrid-engine",
        blocks=(figure,),
        quality=evaluate_parse_quality((figure,), page_count=1, parser="mineru"),
    )

    enriched = await enrich_visual_evidence(document, StubExtractor())

    assert enriched.quality.status == "ready"
    assert enriched.blocks[0].structured_content["visual_analysis_status"] == "ready"
    assert "0.91" in enriched.blocks[0].text
    assert enriched.blocks[0].image_artifact_id == figure.image_artifact_id


def test_hierarchy_retrieves_children_and_preserves_whole_block_overlap() -> None:
    blocks = (
        _block("a", text="alpha " * 160),
        _block("bridge", text="bridge " * 40),
        _block("b", text="beta " * 160),
        _block("c", text="gamma " * 160),
    )
    document = ParsedDocument(
        document_id="pdf:" + "a" * 24,
        source_uri="paper://one",
        sha256="a" * 64,
        bytes_read=100,
        page_count=1,
        parser="native",
        parser_version="test",
        parser_backend="native",
        blocks=blocks,
        quality=evaluate_parse_quality(blocks, page_count=1, parser="native"),
    )

    units = build_parent_child_units(
        document,
        parent_target_tokens=800,
        parent_max_tokens=1_000,
        child_target_tokens=200,
        child_max_tokens=240,
        child_overlap_tokens=60,
    )

    parents = [unit for unit in units if unit.role == "parent"]
    children = [unit for unit in units if unit.role == "child"]
    assert len(parents) == 1
    assert len(children) == 3
    assert children[0].block_ids == ("a", "bridge")
    assert children[1].block_ids == ("bridge", "b")
    assert all(child.parent_id == parents[0].unit_id for child in children)
    assert children[0].next_unit_id == children[1].unit_id
    assert children[1].previous_unit_id == children[0].unit_id


def test_hierarchy_keeps_table_and_caption_in_one_atomic_child() -> None:
    blocks = (
        _block("intro", text="introduction " * 100),
        _block("table", block_type="table", text="Model A recall 0.91"),
        _block("caption", block_type="caption", text="Table 1 retrieval results"),
    )
    document = ParsedDocument(
        document_id="pdf:" + "a" * 24,
        source_uri="paper://one",
        sha256="a" * 64,
        bytes_read=100,
        page_count=1,
        parser="mineru",
        parser_version="test",
        parser_backend="hybrid-engine",
        blocks=blocks,
        quality=evaluate_parse_quality(blocks, page_count=1, parser="mineru"),
    )

    children = [
        unit
        for unit in build_parent_child_units(document)
        if unit.role == "child"
    ]

    assert any(unit.block_ids == ("table", "caption") for unit in children)


def test_flat_ablation_stays_page_bounded_and_has_no_parent_index() -> None:
    blocks = (
        _block("p1-a", text="alpha " * 80, page=1),
        _block("p1-b", text="beta " * 80, page=1),
        _block("p2", text="gamma " * 80, page=2),
    )
    document = ParsedDocument(
        document_id="pdf:" + "a" * 24,
        source_uri="paper://one",
        sha256="a" * 64,
        bytes_read=100,
        page_count=2,
        parser="native",
        parser_version="test",
        parser_backend="native",
        blocks=blocks,
        quality=evaluate_parse_quality(blocks, page_count=2, parser="native"),
    )

    units = build_flat_units(document, target_chars=1_000)

    assert units
    assert all(unit.role == "child" and unit.parent_id == unit.unit_id for unit in units)
    assert all(len(unit.pages) == 1 for unit in units)


def test_flat_overlap_reuses_complete_blocks_without_crossing_pages() -> None:
    blocks = (
        _block("p1-a", text="alpha " * 20, page=1),
        _block("p1-b", text="beta " * 20, page=1),
        _block("p1-c", text="gamma " * 20, page=1),
        _block("p2-a", text="delta " * 20, page=2),
    )
    document = ParsedDocument(
        document_id="pdf:" + "a" * 24,
        source_uri="paper://one",
        sha256="b" * 64,
        bytes_read=100,
        page_count=2,
        parser="mineru",
        parser_version="test",
        parser_backend="hybrid-engine",
        blocks=blocks,
        quality=evaluate_parse_quality(blocks, page_count=2, parser="mineru"),
    )

    units = build_flat_units(document, target_chars=300, overlap_chars=130)

    assert len(units) >= 3
    assert all(len(unit.pages) == 1 for unit in units)
    assert any(
        left.block_ids[-1] == right.block_ids[0]
        for left, right in zip(units, units[1:], strict=False)
        if left.pages == right.pages
    )
    assert not any(
        left.block_ids[-1] == right.block_ids[0]
        for left, right in zip(units, units[1:], strict=False)
        if left.pages != right.pages
    )


def test_sliding_windows_retain_page_and_block_provenance() -> None:
    blocks = (
        _block("p1-a", text="alpha " * 80, page=1),
        _block("p1-b", text="beta " * 80, page=1),
        _block("p2-a", text="gamma " * 80, page=2),
    )
    document = ParsedDocument(
        document_id="pdf:" + "a" * 24,
        source_uri="paper://one",
        sha256="c" * 64,
        bytes_read=100,
        page_count=2,
        parser="mineru",
        parser_version="test",
        parser_backend="hybrid-engine",
        blocks=blocks,
        quality=evaluate_parse_quality(blocks, page_count=2, parser="mineru"),
    )

    units = build_sliding_window_units(
        document,
        window_chars=256,
        overlap_chars=64,
    )

    assert len(units) > 2
    assert all(unit.role == "child" and unit.parent_id == unit.unit_id for unit in units)
    assert all(len(unit.pages) == 1 for unit in units)
    assert any(
        set(left.block_ids).intersection(right.block_ids)
        for left, right in zip(units, units[1:], strict=False)
        if left.pages == right.pages
    )


@pytest.mark.asyncio
async def test_auto_router_uses_mineru_after_unusable_native_parse() -> None:
    class StubParser:
        def __init__(self, name: str, document: ParsedDocument) -> None:
            self.name = name
            self.document = document
            self.calls = 0

        async def parse(self, path: Path, *, source_uri: str) -> ParsedDocument:
            self.calls += 1
            return self.document

    base = dict(
        document_id="pdf:" + "a" * 24,
        source_uri="paper://one",
        sha256="a" * 64,
        bytes_read=100,
        page_count=1,
    )
    native = StubParser(
        "native",
        ParsedDocument(
            **base,
            parser="native",
            parser_version="native-test",
            parser_backend="native",
            blocks=(),
            quality=evaluate_parse_quality((), page_count=1, parser="native"),
        ),
    )
    mineru_block = _block("mineru-evidence", text="OCR recovered evidence")
    mineru_block = mineru_block.model_copy(
        update={"parser": "mineru", "parser_version": "mineru-test"}
    )
    mineru = StubParser(
        "mineru",
        ParsedDocument(
            **base,
            parser="mineru",
            parser_version="mineru-test",
            parser_backend="hybrid-engine",
            blocks=(mineru_block,),
            quality=evaluate_parse_quality(
                (mineru_block,), page_count=1, parser="mineru", ocr_used=True
            ),
        ),
    )

    result = await PDFParserRouter(native, mineru).parse(
        Path("ignored.pdf"), source_uri="paper://one"
    )

    assert result.parser == "mineru"
    assert [attempt.outcome for attempt in result.attempts] == ["rejected", "accepted"]
    assert native.calls == mineru.calls == 1
