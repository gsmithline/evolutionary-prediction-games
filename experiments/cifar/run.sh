#! /bin/bash

#SBATCH --cpus-per-task 16
#SBATCH --gres gpu:1
# --mem 4G

ind=$SLURM_ARRAY_TASK_ID
total=$SLURM_ARRAY_TASK_MAX
flip_prob=$(python -c "print($ind/$total)")
echo flip_prob=$flip_prob
echo hostname=`hostname`

out_file=output-${SLURM_ARRAY_JOB_ID}-${ind}_of_${total}.pkl

script_dir=../evoml-private/cifar
python $script_dir/train_cifar.py --config-file $script_dir/default_config.yaml --training.flip_prob $flip_prob --training.output_file $out_file

# sbatch -a 0-6 --nodelist NODENAME -N 1 ../evoml-private/cifar/run.sh && watch squeue -u $USER
