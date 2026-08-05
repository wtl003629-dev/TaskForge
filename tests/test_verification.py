from __future__ import annotations

import json
import sqlite3

import pytest

from taskforge.verification import (
    SQLiteVerificationStore,
    VerificationRecord,
    VerificationSignatureError,
)


def test_signed_record_binds_signature_to_exact_content() -> None:
    record = VerificationRecord.signed(
        kind="live_smoke",
        provider="deepseek",
        model="deepseek-chat",
        run_id="run-1",
        evidence={"passed": True, "calculator_output": {"value": 43}},
    )
    assert record.verify_signature() is True
    assert record.signature.startswith("sig:")
    assert len(record.signature) == 4 + 64

    tampered = record.model_copy(deep=True)
    tampered.evidence = {"passed": False}
    assert tampered.verify_signature() is False


def test_store_save_and_latest_round_trip(tmp_path) -> None:
    store = SQLiteVerificationStore(tmp_path / "verification.sqlite3")
    store.save(
        VerificationRecord.signed(
            kind="live_smoke",
            provider="deepseek",
            model="deepseek-chat",
            run_id="run-1",
            evidence={"passed": True},
        )
    )
    latest = store.latest("live_smoke", provider="deepseek", model="deepseek-chat")
    assert latest is not None
    assert latest.run_id == "run-1"

    assert (
        store.latest("live_smoke", provider="deepseek", model="other-model")
        is None
    )
    assert store.latest("business_e2e", provider="deepseek") is None


def test_store_rejects_an_invalid_signature_on_save(tmp_path) -> None:
    store = SQLiteVerificationStore(tmp_path / "verification.sqlite3")
    forged = VerificationRecord.signed(
        kind="live_smoke",
        provider="deepseek",
        model="deepseek-chat",
        run_id="run-forged",
        evidence={"passed": True},
    )
    forged.signature = "sig:" + "0" * 64
    with pytest.raises(VerificationSignatureError, match="invalid signature"):
        store.save(forged)


def test_tampered_persisted_record_fails_closed_on_read(tmp_path) -> None:
    store = SQLiteVerificationStore(tmp_path / "verification.sqlite3")
    store.save(
        VerificationRecord.signed(
            kind="live_smoke",
            provider="deepseek",
            model="deepseek-chat",
            run_id="run-1",
            evidence={"passed": True},
        )
    )
    connection = sqlite3.connect(store.path)
    row = connection.execute(
        "SELECT record_json FROM verification_records"
    ).fetchone()[0]
    data = json.loads(row)
    data["run_id"] = "run-HACKED"
    connection.execute(
        "UPDATE verification_records SET record_json = ?", (json.dumps(data),)
    )
    connection.commit()
    connection.close()

    with pytest.raises(VerificationSignatureError, match="tampered"):
        store.all()
