Overlays for the patch grid (`docs/patch_grid/prereg.md`). Each file names a `base` config and the fields that
differ from it; `artifacts/patch_grid_run.py` merges them and checks that nothing else differs from A1.
They are not runnable configs on their own. Every arm is bf16 (`autocast: bf16`), the patch-16 references included.
