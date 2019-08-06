from incense import ExperimentLoader


if __name__ == "__main__":
    N_to_mask = 4
    dataset = "mnist"
    rollouts = 1000
    use_test_data = False
    restricted_first = False

    if restricted_first:
        restricted_query = {"config.second_agent_label": None}
    else:
        restricted_query = {"config.first_agent_label": None}

    with open("mongo.txt") as f:
        MONGO_URL = f.readline().strip()
    loader = ExperimentLoader(mongo_uri=MONGO_URL, db_name="debate")
    query = {
        "$and": [
            {"config.N_to_mask": N_to_mask},
            restricted_query,
            {"config.dataset": dataset},
            {"config.rollouts": rollouts},
        ]
    }
    experiments = loader.find(query)
    print(
        "Found {} experiments in the database with the following properties:".format(
            len(experiments)
        )
    )
    print(query["$and"])
    print()
    print("Example:")
    print(experiments[0].config)
    print("true label", experiments[0].metrics["true_label"][0])
    print("probabilities", experiments[0].result["values"])
