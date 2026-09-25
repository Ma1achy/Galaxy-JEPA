#!/bin/bash
# Usage: run-in-image.sh regress
#        run-in-image.sh stage0 <tag> [extra SpArcFiRe flags...]      (run.sh flags + extras, toys)
#        run-in-image.sh fits <tag> <fits_dir> <elps_dir|NONE> [flags...]  (FITS path; flags as given)
# The SpArcFiRe clone is mounted at /sparcfire, the toys at /toys (read-only), outputs under /out.
set -u
PM=${PITCH_METHODS:-$HOME/Documents/pitch-methods}
SF=${SPARCFIRE_REPO:-$PM/SpArcFiRe}
TOYS=${TOYS_DIR:-$PM/ht_toys/SpArcFiRe-HT-Response/input}
OUTD=${SF_OUT:-$PM/sparcfire-docker/out}
MODE=$1; shift
EXTRA=()
if [ "$MODE" = fits ]; then
  EXTRA=(-v "$2":/in:ro)
  [ "$3" != NONE ] && EXTRA+=(-v "$3":/elps:ro)
fi
docker run --rm --platform linux/amd64 \
  -v "$SF":/sparcfire \
  -v "$TOYS":/toys:ro \
  -v "$OUTD":/out ${EXTRA[@]+"${EXTRA[@]}"} \
  -e USER=bb sparcfire-official:r2017a bash -c '
    export PYTHON=python
    MODE='"$MODE"'
    if [ $MODE = regress ]; then
      cd /sparcfire && ./regression-test-all.sh
    elif [ $MODE = fits ]; then
      TAG=$1; ELPS=$3; shift 3
      rm -rf /out/$TAG; mkdir -p /out/$TAG/in /out/$TAG/tmp /out/$TAG/output
      for f in /in/*.fits; do ln -sf $f /out/$TAG/in/; done
      source /sparcfire/setup.bash /sparcfire >/dev/null 2>&1
      export PATH=/sparcfire/scripts:$PATH
      E=(); [ "$ELPS" != NONE ] && E=(-elps_dir /elps)
      cd /out/$TAG && /usr/bin/time -v SpArcFiRe -convert-FITS "${E[@]}" ./in ./tmp ./output "$@"
    else
      TAG=$1; shift
      mkdir -p /out/$TAG/in /out/$TAG/tmp /out/$TAG/output
      for f in ${LIST:-/toys/*.jpg}; do ln -sf $f /out/$TAG/in/; done
      source /sparcfire/setup.bash /sparcfire >/dev/null 2>&1
      export PATH=/sparcfire/scripts:$PATH
      cd /out/$TAG && /usr/bin/time -v SpArcFiRe -convert-FITS ./in ./tmp ./output \
        -stopThres 0.1 -useImageStandardization 0 -allowArcBeyond2pi 0 -errRatioThres 5 -unsharpMaskSigma 10 "$@"
    fi' _ "$@"
