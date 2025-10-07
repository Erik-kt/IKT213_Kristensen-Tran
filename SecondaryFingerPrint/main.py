import os
from matchers import Config
from evaluate import process_dataset

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATASET = os.path.join(BASE, "data_check")
    RESULTS = os.path.join(BASE, "results_compare")

    cfg = Config()
    process_dataset(DATASET, RESULTS, cfg=cfg)
