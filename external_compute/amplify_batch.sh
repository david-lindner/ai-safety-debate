#!/bin/bash
#
#SBATCH --job-name=amplify_batch
#SBATCH --output=amplify_out.txt
#
#SBATCH --time=24:00:00
#
#SBATCH --array=0-15

EVAL_UNRESTRICTED=('True' 'False')
INDEX_OF_TRUTH_AGENT=(0 1)
START_AT_SAMPLE=(0 25 50 75)

srun -o ~/ai-safety-debate/out_from_experiments/amplify_$SLURM_ARRAY_TASK_ID.out \
     -e ~/ai-safety-debate/out_from_experiments/amplify_$SLURM_ARRAY_TASK_ID.err \
	/bin/bash ~/ai-safety-debate/external_compute/amplify_judge_with_debate.sh \
	${START_AT_SAMPLE[$(($SLURM_ARRAY_TASK_ID % 4))]} \
	25 \
	10000 \
	${EVAL_UNRESTRICTED[$(($SLURM_ARRAY_TASK_ID / 4 % 2))]} \
	${INDEX_OF_TRUTH_AGENT[$(($SLURM_ARRAY_TASK_ID / 4 / 2 % 2))]} \
	'mnist' \
	6 \
	'saved_models/mnist6'
