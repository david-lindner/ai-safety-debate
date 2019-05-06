ml TensorFlow/1.13.1-fosscuda-2018b-Python-3.6.6
source bin/activate
python3 train_classifier_via_debate.py with judge_path=saved_models/mnist4 dataset=mnist rollouts=100 N_epochs=1 batch_size=128 cheat_debate=False classifier_path=saved_models/cheatingClassifier
#python3 train_classifier_via_debate.py with judge_path=saved_models/mnist4 dataset=mnist rollouts=100 N_epochs=1 batch_size=128 cheat_debate=True classifier_path=saved_models/cheatingClassifier
