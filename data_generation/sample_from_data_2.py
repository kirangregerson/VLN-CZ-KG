import json
import os
import random

DATASET_PATH = '/home/grail/VLN-CE-master/data/datasets/RxR_VLNCE_v0_og/train/train_follower.json'
WRITE_PATH = 'rxr_train_guide_small_cutoff.json'
NUM_SAMPLE = 444

def load_dataset(path_to_data):
    f = open(path_to_data)
    episodes = json.load(f)
    f.close()
    return episodes["episodes"]

def select_samples(data):
    """
    Parameters
    ----------
    data : Array[Dictionary]
    """
    seen = set()

    new = {"episodes": []}

    f = open(WRITE_PATH, 'a')

    for i in range(NUM_SAMPLE):
        new["episodes"].append(data[i])

    json.dump(new, f)

    f.close()

def process():
    data = load_dataset(DATASET_PATH)
    select_samples(data)

if __name__ == '__main__':
    process()
