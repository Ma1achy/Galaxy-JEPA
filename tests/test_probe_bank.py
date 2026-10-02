"""Probe banks carry their encoder's SHA-1 and refuse to load against a different encoder.

`artifacts/probe_bank.py` lives in artifacts/ because every bank builder and loader does: nothing in
the package reads a bank back (the second-consumer rule), so it is imported here by path. Synthetic
banks and synthetic "checkpoints" in tmp_path — no encoder, no real bank.

A bank with no `checkpoint_sha1` raises rather than loading under a `legacy=True` flag: the
backfill command is the one legacy path, it stamps the bank as 'backfilled' so the weaker guarantee
is visible on the artefact, and a flag at the call site would be the silent default the invariants
forbid.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))
import probe_bank as pb  # noqa: E402

pytestmark = pytest.mark.invariant


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    p = tmp_path / "encoder.pt"
    p.write_bytes(b"weights of encoder A")
    return p


def _bank(tmp_path: Path, checkpoint, name: str = "bank.npz", **extra) -> Path:
    path = tmp_path / name
    pb.write_bank(
        path,
        ids=np.arange(5, dtype=np.int64),
        x=np.arange(15, dtype=np.float32).reshape(5, 3),
        checkpoint=checkpoint,
        **extra,
    )
    return path


def test_write_records_path_and_sha1(tmp_path, ckpt):
    path = _bank(tmp_path, ckpt)
    with np.load(path, allow_pickle=False) as b:
        assert str(b["checkpoint"]) == str(ckpt)
        assert str(b["checkpoint_sha1"]) == hashlib.sha1(ckpt.read_bytes()).hexdigest()
        assert str(b["checkpoint_provenance"]) == "built"
        assert np.array_equal(b["x"], np.arange(15, dtype=np.float32).reshape(5, 3))
    # uncompressed, so plate_export can still memory-map x; no temporary file left behind
    assert zipfile.ZipFile(path).getinfo("x.npy").compress_type == zipfile.ZIP_STORED
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_load_with_the_matching_checkpoint(tmp_path, ckpt):
    copy = tmp_path / "moved" / "encoder.pt"  # same bytes elsewhere: a location is not an identity
    copy.parent.mkdir()
    copy.write_bytes(ckpt.read_bytes())
    path = _bank(tmp_path, ckpt)
    for expected in (None, ckpt, copy):
        with pb.load_bank(path, expected_checkpoint=expected) as b:
            assert np.array_equal(b["ids"], np.arange(5))


def test_a_different_expected_checkpoint_raises(tmp_path, ckpt):
    other = tmp_path / "other.pt"
    other.write_bytes(b"weights of encoder B")
    path = _bank(tmp_path, ckpt)
    with pytest.raises(pb.BankProvenanceError, match="different encoder"):
        pb.load_bank(path, expected_checkpoint=other)


def test_a_checkpoint_changed_under_the_bank_raises(tmp_path, ckpt):
    path = _bank(tmp_path, ckpt)
    ckpt.write_bytes(b"retrained in place, same path")
    with pytest.raises(pb.BankProvenanceError, match="changed under the bank"):
        pb.load_bank(path)


def test_a_bank_without_the_field_raises_and_names_the_backfill(tmp_path, ckpt):
    path = tmp_path / "legacy.npz"
    np.savez(path, ids=np.arange(3), x=np.zeros((3, 2)), checkpoint=str(ckpt))
    with pytest.raises(pb.BankProvenanceError, match="probe_bank.py backfill"):
        pb.load_bank(path)


def test_untrained_banks_carry_a_descriptor(tmp_path, ckpt):
    config = {"embed_dim": 384, "depth": 12}
    desc = pb.untrained_descriptor(config, 0)
    assert desc.startswith("untrained seed 0, config sha1 ")
    # key order is not identity
    assert pb.untrained_descriptor({"depth": 12, "embed_dim": 384}, 0) == desc
    assert pb.untrained_descriptor(config, 1) != desc
    path = _bank(tmp_path, desc)
    with np.load(path, allow_pickle=False) as b:
        assert str(b["checkpoint_sha1"]) == "" and str(b["checkpoint_provenance"]) == "untrained"
    pb.load_bank(path, expected_checkpoint=desc).close()
    with pytest.raises(pb.BankProvenanceError, match="is for"):
        pb.load_bank(path, expected_checkpoint=pb.untrained_descriptor(config, 1))
    # an untrained bank is not a trained one
    with pytest.raises(pb.BankProvenanceError, match="is for"):
        pb.load_bank(path, expected_checkpoint=ckpt)


def test_reserved_keys_cannot_be_smuggled_in_as_extras(tmp_path, ckpt):
    with pytest.raises(ValueError, match="reserved"):
        _bank(tmp_path, ckpt, checkpoint_sha1="0" * 40)


def test_backfill_adds_the_field_and_leaves_every_member_byte_identical(tmp_path, ckpt):
    path = tmp_path / "legacy.npz"
    x = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
    np.savez(path, ids=np.arange(4), x=x, checkpoint=str(ckpt))
    with zipfile.ZipFile(path) as z:
        before = {i.filename: z.read(i.filename) for i in z.infolist()}

    report = pb.backfill(path, today="2026-09-27")
    assert report.startswith("stamp")
    with zipfile.ZipFile(path) as z:
        assert {n: z.read(n) for n in before} == before
    with pb.load_bank(path, expected_checkpoint=ckpt) as b:
        assert str(b["checkpoint_provenance"]).startswith("backfilled 2026-09-27")
        assert np.array_equal(b["x"], x)
    assert pb.backfill(path).startswith("skip")  # idempotent


def test_backfill_skips_a_bank_that_records_no_checkpoint(tmp_path):
    path = tmp_path / "noise.npz"
    np.savez(path, ids=np.arange(2), x=np.zeros((2, 2)))
    assert "records no checkpoint" in pb.backfill(path)
    with np.load(path) as b:
        assert "checkpoint_sha1" not in b.files


def test_backfill_of_an_untrained_seed_bank_checks_its_architecture_key(tmp_path):
    config = {"embed_dim": 384, "depth": 12}
    path = tmp_path / "seeds.npz"
    np.savez(
        path, ids=np.arange(2), model_key="key-of-A", seed1=np.zeros((2, 2)), seed2=np.ones((2, 2))
    )
    assert "does not match" in pb.backfill(
        path, untrained_config=config, model_key=lambda c: "key-of-B"
    )
    assert pb.backfill(path, untrained_config=config, model_key=lambda c: "key-of-A").startswith(
        "stamp"
    )
    pb.load_bank(path, expected_checkpoint=pb.untrained_descriptor(config, "1+2")).close()
