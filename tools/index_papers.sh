#!/usr/bin/env bash
# Build grep-able text for the agents from PDFs in chimes_papers/ (gitignored).
# Known ChIMES papers get the stable names the chimes-literature skill cites;
# any other PDF keeps its own name. Requires pdftotext/pdfinfo (poppler).
set -euo pipefail
cd "$(dirname "$0")/../chimes_papers"
mkdir -p text
declare -A NAME=(
  [ct7b00867.pdf]=2017_JCTC_Lindsey_ChIMES_molten_carbon
  [ct8b00831.pdf]=2019_JCTC_Lindsey_ChIMES_water
  [134117_1_online.pdf]=2020_JCP_Lindsey_active_learning_reactive
  [224102_1_online.pdf]=2020_JCP_Pham_HN3_detonation
  [144112_1_5.0141616.pdf]=2023_JCP_Goldman_DFTB_ChIMES
  [s41524-024-01497-y.pdf]=2024_npjCM_Lindsey_ChIMES_carbon_2.0
  [s41524-025-01863-4.pdf]=2025_npjCM_Lindsey_hierarchical_transfer_learning
  [ci5c02179.pdf]=2026_JCIM_Laubach_cluster_graph_fingerprinting
)
: > text/INDEX.txt
for f in *.pdf; do
  out="${NAME[$f]:-${f%.pdf}}"
  pdftotext "$f" "text/$out.txt"
  title=$(pdfinfo "$f" 2>/dev/null | sed -n 's/^Title: *//p')
  [ -z "$title" ] && title=$(head -c 400 "text/$out.txt" | tr '\n' ' ' | cut -c1-160)
  printf '%s\t%s\t%s\n' "$out.txt" "$f" "$title" >> text/INDEX.txt
done
echo "indexed $(ls text/*.txt | grep -vc INDEX) papers into chimes_papers/text/"
