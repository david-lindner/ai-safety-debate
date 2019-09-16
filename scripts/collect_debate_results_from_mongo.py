import datetime
from json import JSONEncoder
import jsonlines
from incense import ExperimentLoader

if __name__ == "__main__":
    N_to_mask = 4
    dataset = "mnist"
    rollouts = 1000
    use_test_data = False
    unrestricted_debate = True
    restricted_first = None

    if unrestricted_debate:
        if restricted_first is not None:
            if restricted_first:
                restricted_query = {"config.second_agent_label": None}
            else:
                restricted_query = {"config.first_agent_label": None}
        else:
            restricted_query = {
                "$or": [
                    {"config.first_agent_label": None},
                    {"config.second_agent_label": None},
                ]
            }
    else:
        restricted_query = {}

    with open("mongo.txt") as f:
        MONGO_URL = f.readline().strip()

    loader = ExperimentLoader(mongo_uri=MONGO_URL, db_name="debate")
    query = {
        "$and": [
            {"config.N_to_mask": N_to_mask},
            restricted_query,
            {"config.dataset": dataset},
            {"config.rollouts": rollouts},
            {"status": "COMPLETED"},
            # {"status": {"$ne": "COMPLETED"}},
        ]
    }
    # experiments = loader.find_latest(10)
    experiments = loader.find(query)
    print("Found {} experiments in the database".format(len(experiments)))

    # for exp in experiments:
    #     print(exp.config["sample_id"])

    with jsonlines.open(
        "debate_dump_{}_{}_rollouts_1000.jsonl".format(dataset, N_to_mask, rollouts),
        "w",
        dumps=JSONEncoder(default=str).encode,
    ) as f:
        for ex in experiments:
            ex_dict = ex.to_dict()
            f.write(
                {
                    "config": ex_dict["config"],
                    "result": ex_dict["result"]["values"],
                    "true_label": ex.metrics["true_label"][0],
                }
            )
            del ex_dict