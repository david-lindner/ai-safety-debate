cd ~/ai-safety-debate/ai-safety-debate

ml TensorFlow/1.13.1-foss-2018b-Python-3.6.6
#ml TensorFlow/1.13.1-fosscuda-2018b-Python-3.6.6

#python3 train_judge.py --dataset "mnist" --N-to-mask 4 --train-steps 30000
#python3 train_judge.py --dataset "mnist" --N-to-mask 6 --train-steps 50000
#python3 train_judge.py --dataset "fashion" --N-to-mask 4 --train-steps 30000
#python3 train_judge.py --dataset "fashion" --N-to-mask 6 --train-steps 50000
python3 train_judge.py --dataset "fashion" --N-to-mask 10 --train-steps 10000
