"""The runner's write guard: Finder's .DS_Store is not an entry escaping its directory; anything else is."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))
import ladder_v2_run as R  # noqa: E402


def test_snapshot_ignores_ds_store_but_sees_real_writes(tmp_path: Path) -> None:
    (tmp_path / "rec.json").write_text("{}")
    before = R._snapshot(tmp_path)
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".DS_Store").write_bytes(b"x")
    assert R._snapshot(tmp_path) == before
    (tmp_path / "sub" / "new.json").write_text("{}")
    assert set(R._snapshot(tmp_path)) - set(before) == {str(tmp_path / "sub" / "new.json")}
