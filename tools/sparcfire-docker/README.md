# Official SpArcFiRe in a container (Brief BB0c)

Recipe only. No SpArcFiRe code and no MATLAB Runtime live in this repository.

- **Image:** `sparcfire-official:r2017a`. It is Ubuntu 18.04 on linux/amd64, run under emulation on
  Apple silicon, and follows SpArcFiRe's `UbuntuSetup.sh`.
- **Runtime:** MATLAB Runtime R2017a (9.2), installed to `/pkg/matlab/R2017a`. Download it into
  this directory before building:

  ```
  https://ssd.mathworks.com/supportfiles/downloads/R2017a/deployment_files/R2017a/installers/glnxa64/MCR_R2017a_glnxa64_installer.zip
  1,334,959,803 bytes, SHA-256 a54f04d360e540986c83dec87bb940c6d37a1843574328729f44019f5fcfac1c
  ```

  Then build it: `docker build --platform linux/amd64 -t sparcfire-official:r2017a .`
- **SpArcFiRe:** a clone of `waynebhayes/SpArcFiRe`, mounted at `/sparcfire` at run time. Its
  location is set by `SPARCFIRE_REPO`, defaulting to `~/Documents/pitch-methods/SpArcFiRe`.
- **Validation:** `./run-in-image.sh regress` runs the repo's `regression-test-all.sh`. The verdict,
  recorded in `artifacts/bb_findings.md` §BB0c, is **PARTIAL**:
  - all four SpArcFiRe tests pass;
  - the GALFIT section is untested, because it needs Python ≥ 3.7 and the image has 3.6.
- **Setup note:** `SPARCFIRE_HOME` is set in the image, so `setup.bash` returns before its pip-package
  count. `regression-test-all.sh` puts `scripts/` on `PATH` itself.
