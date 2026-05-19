#!/usr/bin/env bash
# Rounds-budget sweep for the physics-discovery agents.
#
# Fixed axes:
#   model  = claude-opus-4-6
#   critic = claude-opus-4-6
#   noise  = 0.075
# Swept axes:
#   world × seed × max_rounds  where:
#     WORLDS = gravity yukawa fractional dark_matter three_species
#     SEEDS  = 0 1 2
#     ROUNDS = 1 2 4 8 16 32
#
# Output:
#   results/bench_rounds/<timestamp>/<safe_model>/<world>_seed<s>_rounds<r>.{json,txt,stdout.log}
#   results/bench_rounds/<timestamp>/summary.jsonl
#
# After the loop, call scripts/analyze_rounds.py to produce the rounds-vs-metric plots.
#
# Override any axis via env var, e.g.:
#   ROUNDS="1 4 16" WORLDS="gravity" SEEDS="0" ./scripts/round_benchmark.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
else
  echo "WARNING: ${REPO_ROOT}/.env not found — API-key-backed calls will fail." >&2
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set (checked .env and current env)." >&2
  exit 1
fi

MODEL="${MODEL:-claude-opus-4-6}"
WORLDS="${WORLDS:-gravity yukawa fractional dark_matter three_species}"
SEEDS="${SEEDS:-0 1 2}"
ROUNDS="${ROUNDS:-1 2 4 8 16 32}"
NOISE_STD="${NOISE_STD:-0.075}"
CRITIC_MODEL="${CRITIC_MODEL:-claude-opus-4-6}"

read -r -a WORLD_ARR  <<< "${WORLDS}"
read -r -a SEED_ARR   <<< "${SEEDS}"
read -r -a ROUNDS_ARR <<< "${ROUNDS}"

START_CELL="${START_CELL:-1}"

if [[ -n "${RESUME_DIR:-}" ]]; then
  OUT_ROOT="${RESUME_DIR}"
  if [[ ! -d "${OUT_ROOT}" ]]; then
    echo "ERROR: RESUME_DIR=${OUT_ROOT} does not exist." >&2
    exit 1
  fi
  TS="$(basename "${OUT_ROOT}")"
  SUMMARY="${OUT_ROOT}/summary.jsonl"
  keep=$((START_CELL - 1))
  if [[ -f "${SUMMARY}" && ${keep} -ge 0 ]]; then
    head -n "${keep}" "${SUMMARY}" > "${SUMMARY}.tmp" && mv "${SUMMARY}.tmp" "${SUMMARY}"
    echo "Resuming into ${OUT_ROOT}; truncated summary to ${keep} rows, starting at cell ${START_CELL}."
  fi
else
  TS="$(date +%Y%m%d-%H%M%S)"
  OUT_ROOT="${BENCH_ROOT:-results/bench_rounds}/${TS}"
  SUMMARY="${OUT_ROOT}/summary.jsonl"
  mkdir -p "${OUT_ROOT}"
fi

N_TOTAL=$(( ${#WORLD_ARR[@]} * ${#SEED_ARR[@]} * ${#ROUNDS_ARR[@]} ))
safe_model="${MODEL//\//_}"

echo "=========================================="
echo "Rounds benchmark"
echo "  output : ${OUT_ROOT}"
echo "  model  : ${MODEL}   (critic ${CRITIC_MODEL}, noise ${NOISE_STD})"
echo "  worlds : ${WORLDS}"
echo "  seeds  : ${SEEDS}"
echo "  rounds : ${ROUNDS}"
echo "  cells  : ${N_TOTAL}"
echo "=========================================="

if [[ -n "${RESUME_DIR:-}" && -f "${OUT_ROOT}/config.json" ]]; then
  echo "Preserving existing ${OUT_ROOT}/config.json."
else
cat > "${OUT_ROOT}/config.json" <<EOF
{
  "timestamp": "${TS}",
  "model": "${MODEL}",
  "critic": "${CRITIC_MODEL}",
  "noise_std": ${NOISE_STD},
  "worlds": [$(printf '"%s",' "${WORLD_ARR[@]}" | sed 's/,$//')],
  "seeds": [${SEEDS// /, }],
  "rounds": [${ROUNDS// /, }]
}
EOF
fi

cell=0
t_start=$(date +%s)

for world in "${WORLD_ARR[@]}"; do
  for seed in "${SEED_ARR[@]}"; do
    for r in "${ROUNDS_ARR[@]}"; do
      cell=$((cell + 1))
      if [[ ${cell} -lt ${START_CELL} ]]; then
        continue
      fi
      base="${OUT_ROOT}/${safe_model}/${world}_seed${seed}_rounds${r}"
      mkdir -p "$(dirname "${base}")"

      t_cell_start=$(date +%s)
      printf "[%3d/%3d] world=%s seed=%s rounds=%s\n" \
        "${cell}" "${N_TOTAL}" "${world}" "${seed}" "${r}"

      python ScienceAgent/run_discovery.py \
        --model "${MODEL}" \
        --world "${world}" \
        --noise-std "${NOISE_STD}" \
        --noise-seed "${seed}" \
        --max-rounds "${r}" \
        --critic on \
        --critic-model "${CRITIC_MODEL}" \
        --store_output "${base}" \
        --quiet \
        > "${base}.stdout.log" 2>&1
      rc=$?
      t_cell_end=$(date +%s)

      if [[ ${rc} -ne 0 ]]; then
        echo "    FAILED (rc=${rc}) — see ${base}.stdout.log"
      else
        printf "    done in %ds\n" "$((t_cell_end - t_cell_start))"
      fi

      python scripts/append_summary.py \
        --run-json "${base}.json" \
        --summary "${SUMMARY}" \
        --seed "${seed}" \
        --critic on \
        --max-rounds "${r}" \
        --return-code "${rc}" \
        --stdout-log "${base}.stdout.log" \
        || echo "    (summary append failed)"
    done
  done
done

t_end=$(date +%s)
echo "=========================================="
echo "Finished ${cell} cells in $((t_end - t_start))s"
echo "Summary : ${SUMMARY}"
echo "=========================================="

python scripts/analyze_rounds.py --summary "${SUMMARY}" --out-dir "${OUT_ROOT}" \
  || echo "Analysis failed — summary.jsonl is still usable."
