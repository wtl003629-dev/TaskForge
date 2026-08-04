from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from taskforge.eval_datasets import (
    DatasetCatalog,
    DatasetDownloadError,
    DatasetSource,
    download_dataset_source,
    load_dataset_catalog,
)


def source_for(body: bytes, **changes) -> DatasetSource:
    payload = {
        "dataset_id": "fixture",
        "name": "Fixture",
        "homepage": "https://example.test/dataset",
        "license": "CC BY 4.0",
        "commercial_use": True,
        "redistribution": "catalog-only",
        "automated": True,
        "artifacts": [
            {
                "filename": "fixture.json",
                "url": "https://data.example.test/fixture.json",
                "sha256": hashlib.sha256(body).hexdigest(),
                "max_bytes": 1000,
            }
        ],
    }
    payload.update(changes)
    return DatasetSource.model_validate(payload)


def test_download_is_allowlisted_bounded_checksum_verified_and_cached(tmp_path) -> None:
    body = b'{"ok":true}'
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    )
    source = source_for(body)

    first = download_dataset_source(
        source,
        output_dir=tmp_path,
        allowed_hosts=frozenset({"data.example.test"}),
        client=client,
    )[0]
    second = download_dataset_source(
        source,
        output_dir=tmp_path,
        allowed_hosts=frozenset({"data.example.test"}),
        client=client,
    )[0]

    assert first.cached is False
    assert second.cached is True
    assert (tmp_path / "fixture.json").read_bytes() == body
    client.close()


def test_download_fails_closed_for_host_checksum_size_and_noncommercial(tmp_path) -> None:
    body = b"too large"
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    )
    with pytest.raises(DatasetDownloadError, match="allowlisted"):
        download_dataset_source(source_for(body), output_dir=tmp_path, client=client)
    bad_hash = source_for(body)
    bad_hash.artifacts[0].sha256 = "0" * 64
    with pytest.raises(DatasetDownloadError, match="checksum"):
        download_dataset_source(
            bad_hash,
            output_dir=tmp_path,
            allowed_hosts=frozenset({"data.example.test"}),
            client=client,
        )
    too_small = source_for(body)
    too_small.artifacts[0].max_bytes = 2
    with pytest.raises(DatasetDownloadError, match="size limit"):
        download_dataset_source(
            too_small,
            output_dir=tmp_path,
            allowed_hosts=frozenset({"data.example.test"}),
            client=client,
        )
    noncommercial = source_for(body, commercial_use=False)
    with pytest.raises(DatasetDownloadError, match="non-commercial"):
        download_dataset_source(
            noncommercial,
            output_dir=tmp_path,
            allowed_hosts=frozenset({"data.example.test"}),
            client=client,
        )
    client.close()


def test_catalog_loads_and_rejects_duplicate_ids(tmp_path) -> None:
    body = b"{}"
    source = source_for(body)
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({"schema_version": "1.0", "sources": [source.model_dump(mode="json")]}),
        encoding="utf-8",
    )
    assert load_dataset_catalog(path).get("fixture").name == "Fixture"
    with pytest.raises(ValueError, match="unique"):
        DatasetCatalog(sources=[source, source])
