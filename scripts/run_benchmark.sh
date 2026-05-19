#!/usr/bin/env bash
# Benchmark driver for the physics-discovery agents.
#
# Runs the full (model × world × noise × critic × seed) matrix serially,
# writes per-cell logs to results/bench/<timestamp>/<model>/..., and appends
# one summary row per cell to results/bench/<timestamp>/summary.jsonl.
#
# After the loop finishes, calls aggregate_bench.py to produce tables/plots.
#
# Override any axis via env var, e.g.:
#   MODELS="claude-opus-4-7" WORLDS="gravity" SEEDS="0" ./scripts/run_benchmark.sh
#
# PARALLEL>1 is not implemented yet — matrix is 180 cells by default and
# we're leaving room for a later xargs/GNU-parallel wrapper.

set -uo pipefail

# Resolve repo root from this script's location so cwd doesn't matter.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Load API keys from .env into this shell so every child `python` call sees them.
# run_discovery.py also loads .env via python-dotenv, but exporting here makes the
# keys visible to any tool in the pipeline (and surfaces missing-key errors early).
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

MODELS="${MODELS:-claude-opus-4-7 claude-opus-4-6 claude-sonnet-4-5}"
WORLDS="${WORLDS:-gravity yukawa fractional dark_matter three_species}"
NOISE_STDS="${NOISE_STDS:-0.0 0.05}"
CRITIC_MODES="${CRITIC_MODES:-off on}"
SEEDS="${SEEDS:-0 1 2}"
CRITIC_MODEL="${CRITIC_MODEL:-claude-opus-4-7}"
PARALLEL="${PARALLEL:-1}"

if [[ "${PARALLEL}" -ne 1 ]]; then
  echo "PARALLEL=${PARALLEL} requested but parallel mode is not implemented; running serially." >&2
fi

read -r -a MODEL_ARR      <<< "${MODELS}"
read -r -a WORLD_ARR      <<< "${WORLDS}"
read -r -a NOISE_ARR      <<< "${NOISE_STDS}"
read -r -a CRITIC_ARR     <<< "${CRITIC_MODES}"
read -r -a SEED_ARR       <<< "${SEEDS}"

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
  OUT_ROOT="${BENCH_ROOT:-results/bench}/${TS}"
  SUMMARY="${OUT_ROOT}/summary.jsonl"
  mkdir -p "${OUT_ROOT}"
fi

N_TOTAL=$(( ${#MODEL_ARR[@]} * ${#WORLD_ARR[@]} * ${#NOISE_ARR[@]} * ${#CRITIC_ARR[@]} * ${#SEED_ARR[@]} ))

echo "=========================================="
echo "Benchmark"
echo "  output : ${OUT_ROOT}"
echo "  models : ${MODELS}"
echo "  worlds : ${WORLDS}"
echo "  noise  : ${NOISE_STDS}"
echo "  critic : ${CRITIC_MODES}  (critic model: ${CRITIC_MODEL})"
echo "  seeds  : ${SEEDS}"
echo "  cells  : ${N_TOTAL}"
echo "=========================================="

# Log the exact config so runs are reproducible. Skip on resume to preserve the original.
if [[ -n "${RESUME_DIR:-}" && -f "${OUT_ROOT}/config.json" ]]; then
  echo "Preserving existing ${OUT_ROOT}/config.json."
else
cat > "${OUT_ROOT}/config.json" <<EOF
{
  "timestamp": "${TS}",
  "models": [$(printf '"%s",' "${MODEL_ARR[@]}" | sed 's/,$//')],
  "worlds": [$(printf '"%s",' "${WORLD_ARR[@]}" | sed 's/,$//')],
  "noise_stds": [${NOISE_STDS// /, }],
  "critic_modes": [$(printf '"%s",' "${CRITIC_ARR[@]}" | sed 's/,$//')],
  "seeds": [${SEEDS// /, }],
  "critic_model": "${CRITIC_MODEL}"
}
EOF
fi

cell=0
t_start=$(date +%s)

for model in "${MODEL_ARR[@]}"; do
  safe_model="${model//\//_}"
  for world in "${WORLD_ARR[@]}"; do
    for noise in "${NOISE_ARR[@]}"; do
      for critic in "${CRITIC_ARR[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
          cell=$((cell + 1))
          if [[ ${cell} -lt ${START_CELL} ]]; then
            continue
          fi
          base="${OUT_ROOT}/${safe_model}/${world}_noise${noise}_critic${critic}_seed${seed}"
          mkdir -p "$(dirname "${base}")"

          critic_flags=()
          if [[ "${critic}" == "on" ]]; then
            critic_flags=(--use-critic --critic-model "${CRITIC_MODEL}")
          fi

          t_cell_start=$(date +%s)
          printf "[%3d/%3d] model=%s world=%s noise=%s critic=%s seed=%s\n" \
            "${cell}" "${N_TOTAL}" "${model}" "${world}" "${noise}" "${critic}" "${seed}"

          python ScienceAgent/run_discovery.py \
            --model "${model}" \
            --world "${world}" \
            --noise-std "${noise}" \
            --noise-seed "${seed}" \
            --store_output "${base}" \
            --quiet \
            "${critic_flags[@]}" \
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
            --critic "${critic}" \
            --critic-model "${CRITIC_MODEL}" \
            --return-code "${rc}" \
            --stdout-log "${base}.stdout.log" \
            || echo "    (summary append failed)"
        done
      done
    done
  done
done

t_end=$(date +%s)
echo "=========================================="
echo "Finished ${cell} cells in $((t_end - t_start))s"
echo "Summary : ${SUMMARY}"
echo "=========================================="

python scripts/aggregate_bench.py --summary "${SUMMARY}" --out-dir "${OUT_ROOT}" \
  || echo "Aggregation failed — summary.jsonl is still usable."
