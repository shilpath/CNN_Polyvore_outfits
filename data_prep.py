import os, json
from itertools import combinations
from utils import Config


def prepare_data():

    root_dir = Config['root_path']

    # Parse compatibility_train.txt to pairwise_compatibility_train.txt
    meta_file = open(os.path.join(root_dir, 'train.json'),'r')
    meta_json = json.load(meta_file)
    set_to_items = {}
    for element in meta_json:
        set_to_items[element['set_id']] = element['items']

    file_write = open(os.path.join(root_dir, 'pairwise_compatibility_train.txt'),'w')

    with open(os.path.join(root_dir, 'compatibility_train.txt'),'r') as file_read:
        line = file_read.readline()
        while line:
            outfit = line.split()
            comb = list(combinations(list(range(1, len(outfit))), 2))
            for pair in comb:
                set1, idx1 = outfit[pair[0]].split('_')
                set2, idx2 = outfit[pair[1]].split('_')
                file_write.write(outfit[0] + ' ' + set_to_items[set1][int(idx1)-1]['item_id'] + ' ' +set_to_items[set2][int(idx2)-1]['item_id'] +'\n')

            line = file_read.readline()

    # Parse compatibility_valid.txt to pairwise_compatibility_valid.txt
    meta_file = open(os.path.join(root_dir, 'valid.json'),'r')
    meta_json = json.load(meta_file)
    set_to_items = {}
    for element in meta_json:
        set_to_items[element['set_id']] = element['items']

    file_write = open(os.path.join(root_dir, 'pairwise_compatibility_valid.txt'),'w')

    with open(os.path.join(root_dir, 'compatibility_valid.txt'),'r') as file_read:
        line = file_read.readline()
        while line:
            outfit = line.split()
            comb = list(combinations(list(range(1, len(outfit))), 2))
            for pair in comb:
                set1, idx1 = outfit[pair[0]].split('_')
                set2, idx2 = outfit[pair[1]].split('_')
                file_write.write(outfit[0] + ' ' + set_to_items[set1][int(idx1)-1]['item_id'] + ' ' +set_to_items[set2][int(idx2)-1]['item_id'] +'\n')

            line = file_read.readline()

    return 'pairwise_compatibility_train.txt', 'pairwise_compatibility_valid.txt'
