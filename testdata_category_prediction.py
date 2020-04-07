"""
@shilpath
Outfit category prediction on the test data using the model generated from either categorical_model_fine_tuned.py
or categorical_model_custom.py.
"""
import tensorflow as tf
from utils import Config
from tqdm import tqdm
import os.path as osp
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data import DataGeneratorTest, polyvore_dataset
import os


if __name__ == '__main__':
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()

    loaded_model = tf.keras.models.load_model('polyvore_trained_mymodel.hdf5')
    inp_list = []
    inp_list_1 = []
    count = 0
    root_dir = Config['root_path']
    with open(os.path.join(root_dir, 'test_category_hw.txt'),'r') as file_read:
        for line in file_read:
            #if count < 10:
             #   count+=1
            line = file_read.readline()
            inp_list_1.append(line.split("\n")[0])
            inp_list.append(line.split("\n")[0] + '.jpg')

    inp_set = (inp_list, transforms['test'])
    test_gen = DataGeneratorTest(inp_set)

    predicted_output_final = []
    pred = loaded_model.predict_generator(test_gen)

    for item in pred:
        predicted_output_final.append(np.argmax(item))

    meta_file = open(osp.join(root_dir, Config['meta_file']), 'r')
    meta_json = json.load(meta_file)
    id_to_category = {}

    for k, v in tqdm(meta_json.items()):
        id_to_category[k] = v['category_id']

    files = os.listdir(osp.join(root_dir, 'images'))
    X = []; y = []

    for x in files:
        if x[:-4] in id_to_category and x not in inp_list_1:
            X.append(x)
            y.append(int(id_to_category[x[:-4]]))

    le = LabelEncoder()
    y = le.fit_transform(y)

    decoded_pred_op = le.inverse_transform(predicted_output_final)
    decoded_pred_op = [str(i) for i in decoded_pred_op]

    output_labels = []
    for item in inp_list_1:
        output_labels.append(id_to_category[item])

    count = 0
    for i in range(len(output_labels)):
        if decoded_pred_op[i]!= output_labels[i]:
            count+=1

    print('accuracy= ', 100 - count/len(output_labels)*100)

    with open('category_prediction_output.txt', 'w') as file_write:
        file_write.write('image_name' + ' predicted' + ' actual' + '\n')
        for i in range(len(inp_list_1)):
            file_write.write(str(inp_list_1[i]) + " " + str(decoded_pred_op[i]) + " " + str(output_labels[i]) + "\n")

    file_write.close()







