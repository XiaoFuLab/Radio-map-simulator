#!/usr/bin/env bash
# Grid of radio maps: 3 emitters × 3 shadow variances × 3 decorrelation distances (= 27 files).
# Same fixed settings as run.sh for SNR, grid, bandwidth, and PSD basis.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Three options each (edit these arrays to change the sweep)
EMITTERS=(4 6 8)
SHADOW_VARS=(2.0 4.0 8.0)
DECORR_DIST=(50 100 150)

for R in "${EMITTERS[@]}"; do
  for ETA in "${SHADOW_VARS[@]}"; do
    for XC in "${DECORR_DIST[@]}"; do
      echo "============================================================"
      echo "Generating radio map with:"
      echo "  emitters                  (--emitters):                 ${R}"
      echo "  shadow variance           (--shadow-variance):           ${ETA}"
      echo "  decorrelation distance    (--decorrelation-distance):   ${XC}"
      echo "============================================================"
      # Filenames: avoid '.' in basename for ETA (use p instead of decimal in name)
      ETA_STR="${ETA//./p}"
      OUT="bulk_R${R}_eta${ETA_STR}_Xc${XC}.npz"
      python generate_radio_map.py \
        --emitters "${R}" \
        --shadow-variance "${ETA}" \
        --decorrelation-distance "${XC}" \
        --snr 0 \
        --space-x 100 \
        --space-y 100 \
        --bandwidth-length 64 \
        --psd-basis 'g' \
        --save-file "${OUT}"
    done
  done
done

TOTAL=$((${#EMITTERS[@]} * ${#SHADOW_VARS[@]} * ${#DECORR_DIST[@]}))
echo "Done. Generated ${TOTAL} files in ${SCRIPT_DIR}."
