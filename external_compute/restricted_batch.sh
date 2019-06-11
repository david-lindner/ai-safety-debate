#!/bin/bash
#
#SBATCH --job-name=restricted_batch
#SBATCH --output=restricted_batch_out.txt
#
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=120G
#
#SBATCH --array=0-19

INDEX_OF_TRUTH_AGENT=(0 1)
START_AT_SAMPLE=(0 10 20 30 40 50 60 70 80 90)

srun -o ~/ai-safety-debate/out_from_experiments/restricted10_$SLURM_ARRAY_TASK_ID.out \
     -e ~/ai-safety-debate/out_from_experiments/restricted10_$SLURM_ARRAY_TASK_ID.err \
	/bin/bash ~/ai-safety-debate/external_compute/amplify_judge_with_debate.sh \
	${START_AT_SAMPLE[$(($SLURM_ARRAY_TASK_ID % 10))]} \
	10 \
	10000 \
	'False' \
	${INDEX_OF_TRUTH_AGENT[$(($SLURM_ARRAY_TASK_ID / 10 % 2))]} \
	'fashion' \
	10 \
	'saved_models/fashion10'
