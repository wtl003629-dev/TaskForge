"""Fine-tune an encoder-only FlagEmbedding reranker on QASPER groups.

The input JSON is produced by ``prepare_qasper_reranker_finetune.py`` and is
paper-disjoint from the locked validation/test splits.  The default settings
are intentionally conservative; use a CUDA Torch environment for the full
fit and ``--max-steps 1`` as a CPU smoke test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from FlagEmbedding.abc.finetune.reranker import (
    AbsRerankerDataArguments,
    AbsRerankerModelArguments,
    AbsRerankerTrainingArguments,
)
from FlagEmbedding.finetune.reranker.encoder_only.base import EncoderOnlyRerankerRunner


def train(
    model_path: Path | str,
    train_data: Path,
    output_dir: Path,
    *,
    epochs: float = 1.0,
    max_steps: int = -1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    train_group_size: int = 8,
    query_max_len: int = 64,
    passage_max_len: int = 384,
    max_examples: int = 100_000_000,
    gradient_checkpointing: bool = True,
    freeze_encoder: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_args = AbsRerankerModelArguments(
        model_name_or_path=str(model_path),
        model_type="encoder",
        use_fast_tokenizer=True,
    )
    data_args = AbsRerankerDataArguments(
        train_data=[str(train_data)],
        train_group_size=train_group_size,
        query_max_len=query_max_len,
        passage_max_len=passage_max_len,
        max_len=query_max_len + passage_max_len,
        max_example_num_per_dataset=max_examples,
    )
    training_args = AbsRerankerTrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        fp16=False,
        bf16=False,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=0,
    )
    runner = EncoderOnlyRerankerRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    if freeze_encoder:
        # A CPU-friendly calibration mode: keep the XLM-R encoder fixed and
        # train only the sequence-classification head. It is still a genuine
        # domain adaptation checkpoint, but should be reported separately from
        # a full encoder fine-tune.
        for name, parameter in runner.model.named_parameters():
            parameter.requires_grad = any(
                marker in name.casefold() for marker in ("classifier", "score")
            )
    runner.run()
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep this as a string: ``Path`` converts Hugging Face IDs to Windows
    # backslashes (``cross-encoder\\...``), which the Hub validator rejects.
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-group-size", type=int, default=8)
    parser.add_argument("--query-max-len", type=int, default=64)
    parser.add_argument("--passage-max-len", type=int, default=384)
    parser.add_argument("--max-examples", type=int, default=100_000_000)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the classification head (CPU-friendly calibration mode).",
    )
    args = parser.parse_args()
    train(
        args.model_path,
        args.train_data,
        args.output_dir,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        train_group_size=args.train_group_size,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
        max_examples=args.max_examples,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        freeze_encoder=args.freeze_encoder,
    )
    print(f"saved reranker to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
