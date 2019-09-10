"""
Script to query and investigate results from the mongo db.

Allows to filter results by the experiment parameters, time, and if it
was completed or not. Specific filters can be activated/deactivated
by (un-)commenting them.
"""

from datetime import datetime
from incense import ExperimentLoader

if __name__ == "__main__":
    dataset = "mnist"
    N_to_mask = 4
    rollouts = 1000
    min_date = datetime(2019, 7, 24, 22, 40, 0)
    max_date = datetime(2019, 9, 25, 12, 35, 0)

    with open("mongo.txt") as f:
        MONGO_URL = f.readline().strip()

    loader = ExperimentLoader(mongo_uri=MONGO_URL, db_name="debate")
    query = {
        "$and": [
            # select datasets
            {"config.dataset": dataset},
            {"config.N_to_mask": N_to_mask},
            {"config.rollouts": rollouts},
            #
            # unrestriced debate
            {
                "$or": [
                    {"config.first_agent_label": None},
                    {"config.second_agent_label": None},
                ]
            },
            #
            # select specific labels
            # {"config.first_agent_label": 1},
            # {"config.second_agent_label": 3},
            #
            # filter for run status
            {"status": "COMPLETED"},
            # {"status": {"$ne": "COMPLETED"}},  # not completed
            #
            # filter by time
            {"start_time": {"$gt": min_date, "$lt": max_date}},
        ]
    }

    # experiments = loader.find_latest(10)  # only collects most recet results (FASTER!)
    experiments = loader.find(query)  # this can take a while depending on the query
    print("Found {} experiments in the database".format(len(experiments)))

    # Prints some results for every entry that was found
    # can take a long time for a lot of experiments
    # (you might want to disable this if you only want to count results)
    for exp in experiments:
        print("sample", exp.config["sample_id"])
        # print("true_label", exp.metrics["true_label"][0])

        # exp_dict = exp.to_dict()
        # print("result", exp_dict["result"]["values"])

        # print("Output")
        # print(exp_dict["captured_out"])
