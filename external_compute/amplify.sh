ml TensorFlow/1.13.1-foss-2018b-Python-3.6.6 # this one is good for cpus
source bin/activate # get access to the virtual environment, to use sacred

# My extremely unsophisticated way of running multiple similar scripts is to set up the different combinations, 
# uncomment one at a time, and start running it. I hope to find a better way to do this

python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=mnist N_to_mask=4 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=mnist N_to_mask=4 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=mnist N_to_mask=6 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=mnist N_to_mask=6 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=mnist N_to_mask=4 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=mnist N_to_mask=4 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=mnist N_to_mask=6 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=mnist N_to_mask=6 index_of_truth_agent=1

#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=fashion N_to_mask=4 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=fashion N_to_mask=4 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=fashion N_to_mask=6 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=fashion N_to_mask=6 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=fashion N_to_mask=4 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=fashion N_to_mask=4 index_of_truth_agent=1
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=fashion N_to_mask=6 index_of_truth_agent=0
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=fashion N_to_mask=6 index_of_truth_agent=1


#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=mnist N_to_mask=4 index_of_truth_agent=0 changing_sides=False
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=mnist N_to_mask=4 index_of_truth_agent=1 changing_sides=False
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=mnist N_to_mask=6 index_of_truth_agent=0 changing_sides=False
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=True dataset=mnist N_to_mask=6 index_of_truth_agent=1 changing_sides=False
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=mnist N_to_mask=4 index_of_truth_agent=0 changing_sides=False
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=mnist N_to_mask=4 index_of_truth_agent=1 changing_sides=False
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=mnist N_to_mask=6 index_of_truth_agent=0 changing_sides=False
#python3 amplify_judge_with_debate.py with rollouts=100 eval_unrestricted=False dataset=mnist N_to_mask=6 index_of_truth_agent=1 changing_sides=False
