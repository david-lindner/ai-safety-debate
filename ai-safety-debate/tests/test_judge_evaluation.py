from run_debate import run
import numpy as np
import pytest

def test_restricted_debate():
    """Run restricted debate, make sure it doesn't crash."""
    liars_label = np.random.randint(10)
    id = np.random.randint(10)
    run(N_to_mask=4, sample_id=id, lying_agent_label=liars_label, judge_path=None, dataset="mnist", rollouts=100)

def test_unrestricted_debate():
    """Run debate with unrestricted liar, make sure it doesn't crash."""
    id = np.random.randint(10)
    run(N_to_mask=4, sample_id=id, lying_agent_label=None, judge_path=None, dataset="mnist", rollouts=100)

def test_nobody_commits():
    """Make sure that if nobody commits, the debate crashes."""
    print('This test isnt implemented yet.')