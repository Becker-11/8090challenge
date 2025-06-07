#!/usr/bin/env bash
# Black Box Challenge – ACME Reimbursement Replica
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>
# Outputs ONE number (2‑decimal float) and nothing else.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <days> <miles> <receipts>" >&2
  exit 1
fi

python3 "$(dirname "$0")/calculate_reimbursement.py" "$1" "$2" "$3"
