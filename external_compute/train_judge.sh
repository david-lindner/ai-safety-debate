ml TensorFlow/1.13.1-fosscuda-2018b-Python-3.6.6 # this one is good for gpus

#python3 train_judge.py --dataset "fashion" --N-to-mask 4 --train-steps 30000
python3 train_judge.py --dataset "fashion" --N-to-mask 6 --train-steps 50000
