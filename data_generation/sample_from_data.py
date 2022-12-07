import json
import os
import random

DATASET_PATH = '/home/grail/VLN-CE-master/data/datasets/RxR_VLNCE_v0/train/train_guide.json'
WRITE_PATH = 'rxr_train_small.json'
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
        random_int = random.randint(0, 444)
        while (data[random_int]['scene_id'], data[random_int]['instruction']['instruction_id']) not in seen:
            random_int = random.randint(0, 444)
        new["episodes"].append(data[random_int])
        seen.add((data[random_int]['scene_id'], data[random_int]['instruction']['instruction_id']))

    json.dump(new, f)

    f.close()

def process():
    data = load_dataset(DATASET_PATH)
    select_samples(data)

if __name__ == '__main__':
    process()
