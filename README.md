# Training classifiers via Debate

## Note: This code is work in progress and is guaranteed to contain bugs.

This repository provides code to reproduce the experiments from [AI Safety via Debate](https://arxiv.org/abs/1805.00899) ([blogpost](https://openai.com/blog/debate/)).

On top of that we run additional experiment on MNIST as well as FashionMNIST data and train classifiers from debate results.

## Setup

Install the python dependencies by running the following in a python 3.6

```
pip install -r requirements.txt
```

## Usage

All code is located in the `ai-safety-debate` folder.

- To train a judge use `train_judge.py`
- To run individual debates use `run_debate.py`
- To evaluate the accuracy of a judge combined with debate use `amlify_judge_with_debate.py`
- To use debate to train a classifier use `train_classifier_via_debate.py`

We use `sacred` for tracking experiments. The results are typically stored in the `experiments` and `amplification_experiments` folders.

## Reproducing 'AI Safety via Debate'

The commands for reproducing the results in table 2 of the [original AI Safety via Debate paper](https://arxiv.org/abs/1805.00899) can be found in `external_compute/amplify.sh`.


## Testing

We use `pytest` for testing (however, not many tests are implemented yet). Run the tests with
```
pytest .
```
