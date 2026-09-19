"""Property tests for the config base + provenance (core/config.py).

Pins: auto-capture round-trip including a *nested* config and a *class reference*;
``*args``/``**kwargs`` rejection at class-definition time; config-hash stability across
a round-trip of a nested tree; and the run-stamp / artefact-stamp writer.
"""

import json
import subprocess
from pathlib import Path

import pytest

from galaxy_jepa.core.config import (
    STAMP_SCHEME,
    Configurable,
    RunStamp,
    class_ref,
    code_sha,
    config_hash,
    resolve_ref,
    write_stamp,
)
from galaxy_jepa.core.gates import MetricGate


# Module-scope so the ClassRef in `Outer.tag` resolves back via import.
class Inner(Configurable):
    def __init__(self, width: int = 4):
        self.width = width


class Outer(Configurable):
    def __init__(self, inner: Inner, label: str = "x", tag: type = Inner):
        self.inner = inner
        self.label = label
        self.tag = tag


def test_classref_round_trip():
    assert resolve_ref(class_ref(MetricGate)) is MetricGate


def test_nested_config_round_trip():
    original = Outer(Inner(width=8), label="y")
    config = original.to_config()
    restored = Outer.from_config(config)
    assert isinstance(restored, Outer)
    assert restored.inner.width == 8
    assert restored.label == "y"
    assert restored.tag is Inner  # class reference survived


def test_config_hash_stable_across_round_trip():
    original = Outer(Inner(width=8), label="y")
    config = original.to_config()
    restored = Outer.from_config(config)
    assert config_hash(config) == config_hash(restored.to_config())


def test_config_hash_changes_with_value():
    a = Outer(Inner(width=8)).to_config()
    b = Outer(Inner(width=9)).to_config()
    assert config_hash(a) != config_hash(b)


def test_varargs_rejected_at_definition():
    with pytest.raises(TypeError):

        class BadArgs(Configurable):
            def __init__(self, *args):
                self.args = args

    with pytest.raises(TypeError):

        class BadKwargs(Configurable):
            def __init__(self, **kwargs):
                self.kwargs = kwargs


def test_unserialisable_value_fails_loudly():
    class Holder(Configurable):
        def __init__(self, payload=None):
            self.payload = payload

    with pytest.raises(TypeError):
        Holder(payload=object()).to_config()


def test_code_sha_returns_sha_and_dirty_flag():
    sha, dirty = code_sha()
    assert isinstance(dirty, bool)
    assert sha == "nogit" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


@pytest.mark.invariant
def test_code_sha_is_read_once_at_process_start_not_at_stamp_time():
    """The stamp must describe the code that RAN, not the tree at the moment it was written.

    ``harness._make_stamp`` is called after ``_prepare``, which can spend an hour baking the
    cache, and Brief M's run spans two days. If dirtiness were sampled there, an edit made while
    a run was underway would be recorded as though the run had executed it — and the artefact
    would be quietly wrong about its own provenance, which is the one thing a stamp exists to
    prevent.

    So the reading is taken at import and cached. This pins both halves: a later call cannot
    re-read the tree, and it cannot be fooled by dirtying it.
    """
    first = code_sha()

    marker = Path(__file__).resolve().parent.parent / "__dirty_probe__.tmp"
    marker.write_text("makes the working tree dirty\n")
    try:
        assert code_sha() == first, "code_sha re-read the tree after it was dirtied"
        # and the raw reading really would have moved -- otherwise the check above proves nothing
        raw = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(marker.parent),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "__dirty_probe__" in raw, "the probe did not actually dirty the tree"
    finally:
        marker.unlink(missing_ok=True)


def test_run_stamp_and_writer(tmp_path):
    config = Outer(Inner(width=8), label="z").to_config()
    stamp = RunStamp.create(config, data_snapshot="manifest:abc123", seed=7)
    # The stamped hash carries its scheme marker; the bare `config_hash` does not, because
    # `data.cache.pipeline_hash` reuses it as the fp16 cache directory name.
    assert stamp.config_hash == STAMP_SCHEME + config_hash(config)
    assert not config_hash(config).startswith(STAMP_SCHEME)
    assert stamp.seed == 7
    assert stamp.data_snapshot == "manifest:abc123"

    stamp_path = write_stamp(stamp, tmp_path / "run", config)
    assert stamp_path.exists()
    written = json.loads(stamp_path.read_text())
    assert written["config_hash"] == STAMP_SCHEME + config_hash(config)
    assert (tmp_path / "run" / "config.json").exists()
