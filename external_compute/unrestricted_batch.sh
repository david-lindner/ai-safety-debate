#!/bin/bash
#
#SBATCH --job-name=unrestricted_batch
#SBATCH --output=unrestricted_batch_out.txt
#
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=70G
#
#SBATCH --array=0-7

INDEX_OF_TRUTH_AGENT=(0 1)
START_AT_SAMPLE=(0 25 50 75)

srun -o ~/ai-safety-debate/out_from_experiments/unrestricted10_$SLURM_ARRAY_TASK_ID.out \
     -e ~/ai-safety-debate/out_from_experiments/unrestricted10_$SLURM_ARRAY_TASK_ID.err \
	/bin/bash ~/ai-safety-debate/external_compute/amplify_judge_with_debate.sh \
	${START_AT_SAMPLE[$(($SLURM_ARRAY_TASK_ID % 4))]} \
	25 \
	10000 \
	true \
	${INDEX_OF_TRUTH_AGENT[$(($SLURM_ARRAY_TASK_ID / 4 % 2))]} \
	'fashion' \
	10 \
	'saved_models/fashion10'
