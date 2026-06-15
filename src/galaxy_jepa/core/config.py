"""Config base + provenance — the experiment record.

Implements ``docs/architecture.md`` "Config is the experiment record" and
``docs/spec/config.md``.

Two layers:

* :class:`Configurable` — the **auto-capture** base. A component derives its config
  from its own ``__init__`` signature (``DECISIONS``-style "configuration over runtime
  cleverness"): kwargs are captured at construction, ``*args``/``**kwargs`` are
  forbidden (reproducible-by-construction), and :meth:`Configurable.to_config` /
  :meth:`from_config` round-trip the component — *recursively*, so a config that nests
  another config (model inside objective inside run) serialises whole, and class
  references travel as a :class:`ClassRef` (a module-path string), not an unpicklable
  object.
* :class:`RunConfig` — a pydantic base for the typed top-level run config, validated at
  load.

A run is fully determined by ``(config hash, code SHA, data-snapshot version, seed,
escape_hatches_used)`` — :class:`RunStamp` — and every artefact is stamped with it via
:func:`write_stamp`, so the headline figures regenerate from their provenance.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import inspect
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

# --- class references --------------------------------------------------------------

_CLASSREF_KEY = "__classref__"
_CONFIG_CLASS_KEY = "__class__"
_CONFIG_FIELDS_KEY = "config"


def class_ref(cls: type) -> str:
    """Serialise a class as a ``"module:qualname"`` reference."""
    return f"{cls.__module__}:{cls.__qualname__}"


def resolve_ref(ref: str) -> type:
    """Resolve a ``"module:qualname"`` reference back to the class."""
    module_path, _, qualname = ref.partition(":")
    if not qualname:
        raise ValueError(f"Malformed class reference {ref!r}; expected 'module:qualname'")
    obj: Any = importlib.import_module(module_path)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


# --- recursive serialisation -------------------------------------------------------


def _serialise(value: Any) -> Any:
    """Recursively serialise a config value to a JSON-safe structure.

    Handles nested :class:`Configurable` objects, class references, pydantic models,
    and the usual containers/primitives. Anything else raises — a config must be
    serialisable by construction, never silently stringified (fail loudly).
    """
    if isinstance(value, Configurable):
        return value.to_config()
    if isinstance(value, type):
        return {_CLASSREF_KEY: class_ref(value)}
    if isinstance(value, BaseModel):
        return {
            _CLASSREF_KEY: class_ref(type(value)),
            "__pydantic__": value.model_dump(mode="json"),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialise(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialise(v) for k, v in value.items()}
    raise TypeError(
        f"Cannot serialise config value of type {type(value).__name__!r}: {value!r}. "
        "Config values must be primitives, containers, class references, Configurables, "
        "or pydantic models."
    )


def _deserialise(value: Any) -> Any:
    """Inverse of :func:`_serialise`."""
    if isinstance(value, dict):
        if _CLASSREF_KEY in value and "__pydantic__" in value:
            model_cls = cast("type[BaseModel]", resolve_ref(value[_CLASSREF_KEY]))
            return model_cls.model_validate(value["__pydantic__"])
        if _CLASSREF_KEY in value and len(value) == 1:
            return resolve_ref(value[_CLASSREF_KEY])
        if _CONFIG_CLASS_KEY in value and _CONFIG_FIELDS_KEY in value:
            cls = cast("type[Configurable]", resolve_ref(value[_CONFIG_CLASS_KEY]))
            return cls.from_config(value)
        return {k: _deserialise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialise(v) for v in value]
    return value


# --- the auto-capture base ---------------------------------------------------------


class Configurable:
    """Base for components whose config is derived from their ``__init__`` signature.

    Subclasses must not declare ``*args``/``**kwargs`` on ``__init__`` — that is
    rejected at class-definition time so a component is always reconstructible from its
    captured kwargs.
    """

    _captured_config: dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only wrap classes that define their own __init__; otherwise the inherited
        # (already-wrapped) __init__ does the capture and we must not double-wrap.
        if "__init__" not in cls.__dict__:
            return
        original_init = cls.__init__
        signature = inspect.signature(original_init)
        for param in signature.parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(
                    f"{cls.__qualname__}.__init__ declares *args/**kwargs ({param.name!r}); "
                    "Configurable components must have an explicit, fixed signature so "
                    "their config is reproducible by construction."
                )

        @functools.wraps(original_init)
        def _capturing_init(self: Configurable, *args: Any, **kw: Any) -> None:
            bound = signature.bind(self, *args, **kw)
            bound.apply_defaults()
            captured = {k: v for k, v in bound.arguments.items() if k != "self"}
            original_init(self, *args, **kw)
            object.__setattr__(self, "_captured_config", captured)

        cls.__init__ = _capturing_init  # type: ignore[assignment,method-assign]

    def to_config(self) -> dict[str, Any]:
        """Return the JSON-safe, recursively-serialised config for this instance."""
        if not hasattr(self, "_captured_config"):
            raise RuntimeError(
                f"{type(self).__qualname__} was constructed without capturing its config; "
                "ensure its __init__ chain runs the Configurable-wrapped __init__."
            )
        return {
            _CONFIG_CLASS_KEY: class_ref(type(self)),
            _CONFIG_FIELDS_KEY: {k: _serialise(v) for k, v in self._captured_config.items()},
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Configurable:
        """Reconstruct an instance from :meth:`to_config` output."""
        target = resolve_ref(config[_CONFIG_CLASS_KEY])
        kwargs = {k: _deserialise(v) for k, v in config[_CONFIG_FIELDS_KEY].items()}
        return target(**kwargs)


def config_hash(config: dict[str, Any]) -> str:
    """Stable sha256 over the canonical JSON of a (recursively-serialised) config tree."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- typed run config --------------------------------------------------------------


class RunConfig(BaseModel):
    """Base for the typed, validated top-level run config (loaded from YAML).

    ``extra='forbid'`` makes an unknown key a loud load-time error rather than a
    silently-ignored typo.
    """

    model_config = ConfigDict(extra="forbid")


# --- provenance --------------------------------------------------------------------

_NO_GIT = "nogit"


def code_sha(repo: Path | None = None) -> tuple[str, bool]:
    """Return ``(git HEAD sha, working-tree-dirty)``.

    Outside a git repo, returns ``("nogit", True)`` with a loud warning rather than a
    hard error — a run can proceed un-versioned, but its irreproducibility is recorded,
    never hidden.
    """
    cwd = str(repo) if repo is not None else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.warn(
            "Not a git repository (or git unavailable); run provenance cannot record a "
            "code SHA. The artefact will be stamped 'nogit' and flagged dirty.",
            stacklevel=2,
        )
        return _NO_GIT, True


@dataclasses.dataclass(frozen=True)
class RunStamp:
    """The provenance tuple stamped onto every artefact.

    ``data_snapshot`` is the data layer's manifest hash (object IDs + the pull query),
    so reproducibility is structural rather than a hand-bumped version string.
    """

    config_hash: str
    code_sha: str
    code_dirty: bool
    data_snapshot: str
    seed: int
    escape_hatches_used: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def create(
        cls,
        config: dict[str, Any],
        *,
        data_snapshot: str,
        seed: int,
        escape_hatches_used: list[str] | None = None,
        repo: Path | None = None,
    ) -> RunStamp:
        sha, dirty = code_sha(repo)
        return cls(
            config_hash=config_hash(config),
            code_sha=sha,
            code_dirty=dirty,
            data_snapshot=data_snapshot,
            seed=seed,
            escape_hatches_used=list(escape_hatches_used or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def write_stamp(
    stamp: RunStamp, artefact_dir: Path, config: dict[str, Any] | None = None
) -> Path:
    """Write ``stamp.json`` (and optionally the full ``config.json``) into an artefact dir.

    Returns the path to the written ``stamp.json``. The directory is created if needed.
    """
    artefact_dir = Path(artefact_dir)
    artefact_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = artefact_dir / "stamp.json"
    stamp_path.write_text(json.dumps(stamp.to_dict(), indent=2, sort_keys=True) + "\n")
    if config is not None:
        (artefact_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
    return stamp_path
