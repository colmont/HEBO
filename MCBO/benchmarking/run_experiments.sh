# Correct modules for AIG/MIG optimization
module load gcc/8.2.0
module load readline/7.0

SEEDS="42 43 44 45 46 47 48 49 50 51"
ABSOLUT_EXE="/cluster/home/cdoumont/HEBO/MCBO/libs/Absolut/src/AbsolutNoLib"
tasks="ackley pest rna_inverse_fold aig_optimization antibody_design mig_optimization"

# Main loop
for task in $tasks; do
    acq_opt="ga"
    tr="basic"
    acq_func="ei"
    model="gp_duv"
    opt_id="${model}__${acq_opt}__${acq_func}__${tr}"
    cmd="python ./experiments/run_task_exps.py --device_id 0 --absolut_dir $ABSOLUT_EXE --task_id $task --optimizers_ids $opt_id --seeds $SEEDS --verbose 2 --variant 3"
    $cmd
done