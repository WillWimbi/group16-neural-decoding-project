#!/bin/bash
#SBATCH --job-name=neural_decoding
#SBATCH -A csds312
#SBATCH -p markov_gpu
#SBATCH --output=slurm_logs/neural_decoding_%j.out
#SBATCH --error=slurm_logs/neural_decoding_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


set -uo pipefail

cd "${SLURM_SUBMIT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/logs" "${RESULTS_DIR}/figures"

echo "=== Job started at $(date) ==="
echo "Results directory: ${RESULTS_DIR}"

source venv/bin/activate

run_script() {
    local script="$1"
    local logname="${script%.py}"
    echo ""
    echo "── Running ${script} ──"
    python "${script}" 2>&1 | tee "${RESULTS_DIR}/logs/${logname}.log"
}

run_script kalman_decoding.py
run_script presentation_plots.py
run_script autoencoder_sweep.py

echo ""
echo "── Collecting outputs ──"

for png in kalman_decoding_results.png \
           fig1_qualitative_check.png \
           fig2_region_check.png \
           fig3_error_histograms.png \
           fig4_results_grid.png \
           fig5_compression.png \
           fig6_sidebyside.png \
           fig7_decoder_vs_reconstruction.png; do
    if [[ -f "$png" ]]; then
        mv "$png" "${RESULTS_DIR}/figures/"
        echo "  moved $png"
    else
        echo "  WARNING: $png not found"
    fi
done

if [[ -d autoencoder_outputs ]]; then
    mv autoencoder_outputs "${RESULTS_DIR}/"
    echo "  moved autoencoder_outputs/"
fi

echo ""
echo "=== Job complete at $(date) ==="
echo "All results in: ${RESULTS_DIR}"
