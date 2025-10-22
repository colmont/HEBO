#!/bin/bash

# sbatch params
cpus="1"
time="1:00:00"
mem_per_cpu="1G"

# conda env
source /home/miniconda3/etc/profile.d/conda.sh
conda activate mcbo_env

run_experiment() {
    local seeds="$1"
    local tasks="$2"
    local opt_ids="$3"
    local perm_invs="$4"

    for OPT_ID in "${opt_ids[@]}"; do
        for SEED in $seeds; do
            for TASK in $tasks; do
                for FLIP in True False; do
                    for PERM_INV in $perm_invs; do
                        cmd="python main.py --gin-files configs/default.gin --seed $SEED --gin-bindings \"Experiments.task_id = \
                            \\\"$TASK\\\"\" \"FLIP = $FLIP\" \"Experiments.opt_id = \\\"$OPT_ID\\\"\" \
                            \"HeatKernel.perm_inv = $PERM_INV\" \"Experiments.debug = False\""
                        sbatch --wrap="$cmd" --cpus-per-task=$cpus --time=$time --mem-per-cpu=$mem_per_cpu
                    done
                done
            done
        done
    done
}

SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61"

# main tasks
TASKS_MAIN="pest contamination labs maxsat60 maxsat125"

# other tasks 
TASKS_APP="rna_inverse_fold antibody_design aig_optimization mig_optimization"

# methods
OPT_IDS_METHODS=(
  "gp_to__is__ei__basic"  # CASMOPOLITAN
  "gp_hed__is__ei__none"  # BODi
  "gp_o__mab__ei__none"  # CoCaBO
  "gp_ssk__ga__ei__none"  # BOSS
  "gp_diff__ls__ei__none" # COMBO
  "gp_rd__mp__addlcb__none"  # RDUCB
)
run_experiment "$SEEDS" "$TASKS_MAIN" "${OPT_IDS_METHODS[@]}" "None"

# kernels
OPT_IDS_KERNELS=(
  "gp_heat__ga__ei__basic"  # Heat (reparam kondor)
  "gp_heat__ga__ei__basic"  # Heat (reparam casmo)
  "gp_oh_rbf__ga__ei__basic"  # One-hot RBF
  "gp_oh_matern__ga__ei__basic"  # One-hot Matern
  "gp_oh_rq__ga__ei__basic"  # One-hot Rational Quadratic 
  "gp_eigen__ga__ei__basic" # COMBO kernel
  "gp_eigen_matern__ga__ei__basic" # Graph Matern
  "gp_hed__ga__ei__basic"  # HED
  "gp_ssk__ga__ei__basic"  # SSK
  "gp_o__ga__ei__basic"  # overlap
  "gp_rd__ga__ei__basic"  # Rand. decomp. (RDUCB)
  "lr_sparse_hs__mab__pi__none"  # Random search
)
run_experiment "$SEEDS" "$TASKS_MAIN" "${OPT_IDS_KERNELS[@]}" "None"

# permutation invariance
TASKS_PERM_INV="ackley schwefel rastrigin sphere styblinski_tang"
run_experiment "$SEEDS" "$TASKS_PERM_INV" "${OPT_IDS_KERNELS[@]}" "None"
run_experiment "$SEEDS" "$TASKS_PERM_INV" "gp_heat__ga__ei__basic" "proj pad_proj"