import json
import os
import random

DATASET_PATH = '/home/grail/VLN-CE-master/data/datasets/RxR_VLNCE_shortened/train/train_guide_gt.json'
WRITE_PATH = 'rxr_train_guide_small_gt.json'
NUM_SAMPLE = 444

def load_dataset(path_to_data):
    f = open(path_to_data)
    episodes = json.load(f)
    f.close()
    return episodes

def select_samples(data):
    """
    Parameters
    ----------
    data : [Dictionary]
    """

    new = {}
    keys = list(data.keys())

    f = open(WRITE_PATH, 'a')

    for i in range(NUM_SAMPLE):
        new[keys[i]] = data[keys[i]]

    json.dump(new, f)

    f.close()

def process():
    data = load_dataset(DATASET_PATH)
    select_samples(data)

if __name__ == '__main__':
    process()
