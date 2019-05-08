cd ~/ai-safety-debate/ai-safety-debate
ml TensorFlow/1.13.1-foss-2018b-Python-3.6.6
source bin/activate
python3 amplify_judge_with_debate.py with start_at_sample=$1 nmbr_samples=$2 rollouts=$3 eval_unrestricted=$4 index_of_truth_agent=$5 dataset=$6  N_to_mask=$7 judge_path=$8

#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=mnist N_to_mask=4 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=mnist N_to_mask=4 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=mnist N_to_mask=6 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=mnist N_to_mask=6 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=mnist N_to_mask=4 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=mnist N_to_mask=4 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=mnist N_to_mask=6 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=mnist N_to_mask=6 index_of_truth_agent=1

#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=fashion N_to_mask=4 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=fashion N_to_mask=4 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=fashion N_to_mask=6 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=fashion N_to_mask=6 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=fashion N_to_mask=4 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=fashion N_to_mask=4 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=fashion N_to_mask=6 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=fashion N_to_mask=6 index_of_truth_agent=1

#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=mnist N_to_mask=4 index_of_truth_agent=0 changing_sides=False
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=mnist N_to_mask=4 index_of_truth_agent=1 changing_sides=False
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=mnist N_to_mask=6 index_of_truth_agent=0 changing_sides=False
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=1000 rollouts=1000 eval_unrestricted=True dataset=mnist N_to_mask=6 index_of_truth_agent=1 changing_sides=False
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=mnist N_to_mask=4 index_of_truth_agent=0 changing_sides=False
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=mnist N_to_mask=4 index_of_truth_agent=1 changing_sides=False
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=mnist N_to_mask=6 index_of_truth_agent=0 changing_sides=False
#python3 amplify_judge_with_debate.py with start_at_sample=0 nmbr_samples=100 rollouts=1000 eval_unrestricted=False dataset=mnist N_to_mask=6 index_of_truth_agent=1 changing_sides=False
