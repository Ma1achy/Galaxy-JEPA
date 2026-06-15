"""Property tests for the config base + provenance (core/config.py).

Pins: auto-capture round-trip including a *nested* config and a *class reference*;
``*args``/``**kwargs`` rejection at class-definition time; config-hash stability across
a round-trip of a nested tree; and the run-stamp / artefact-stamp writer.
"""

import json

import pytest

from galaxy_jepa.core.config import (
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


def test_run_stamp_and_writer(tmp_path):
    config = Outer(Inner(width=8), label="z").to_config()
    stamp = RunStamp.create(config, data_snapshot="manifest:abc123", seed=7)
    assert stamp.config_hash == config_hash(config)
    assert stamp.seed == 7
    assert stamp.data_snapshot == "manifest:abc123"

    stamp_path = write_stamp(stamp, tmp_path / "run", config)
    assert stamp_path.exists()
    written = json.loads(stamp_path.read_text())
    assert written["config_hash"] == config_hash(config)
    assert (tmp_path / "run" / "config.json").exists()
