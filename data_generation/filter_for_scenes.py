import json
import os

DATASET_PATH = '../rxr_train_guide.jsonl'
WRITE_PATH = 'rxr_train_new'
SCANS_DIR = '/home/grail/VLN-CZ-KG/data/scene_datasets/mp3d/v1/scans'

def load_dataset(path_to_data):
    new_data = []
    for line in open(path_to_data, encoding='utf8'):
        new_data.append(json.loads(line))
    return new_data

def generate_filtered_by_scene(data, write_to_path):
    f = open(write_to_path, 'a', encoding='utf8')
    scenes = get_list_of_scenes()
    filtered_data = filter_by_scenes(data, scenes)
    for annotation in filtered_data:
        json.dump(annotation, f, ensure_ascii=False)
    f.close()
    print('done writing')

def get_list_of_scenes():
    return [subdirectory[0][len(SCANS_DIR) + 1:] for subdirectory in os.walk(SCANS_DIR)][1:]

def filter_by_scenes(data, scenes):
    scenes_set = set(scenes)

    return list(filter(lambda annotation : annotation["scan"] in scenes_set, data))

def process():
    data = load_dataset(DATASET_PATH)
    generate_filtered_by_scene(data, WRITE_PATH)

if __name__ == '__main__':
    process()
