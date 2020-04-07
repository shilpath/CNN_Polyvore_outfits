from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image
import tensorflow
from utils import Config


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()


    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms


    def create_dataset(self):
        # map id to category
        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v['category_id']

        # create X, y pairs
        files = os.listdir(self.image_dir)
        X = []; y = []
        inp_list = []
        with open(osp.join(self.root_dir, 'test_category_hw.txt'), 'r') as file_read:
            for line in file_read:
                line = file_read.readline()
                inp_list.append(line.split("\n")[0] + '.jpg')

        for x in files:
            if x[:-4] in id_to_category and x not in inp_list:
                X.append(x)
                y.append(int(id_to_category[x[:-4]]))

        y = LabelEncoder().fit_transform(y)
        print('len of X: {}, # of categories: {}'.format(len(X), max(y) + 1))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, max(y) + 1


    def create_compatibility_dataset2(self, filename):
        meta_file = open(osp.join(self.root_dir, filename), 'r')
        x_train_1 = []
        y_train = []
        for line in meta_file:
            y_train.append(float(line.split(" ")[0]))
            x_train_1.append([line.split(" ")[1] + ".jpg", line.split(" ")[2].split("\n")[0] + ".jpg"])

        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v['category_id']

        return x_train_1, y_train


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = params['n_classes']
        self.X, self.y, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        X, y = np.stack(X), np.stack(y)
        return np.moveaxis(X, 1, 3), tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __data_generation(self, indexes):
        X = []; y = []
        for idx in indexes:
            file_path = osp.join(self.image_dir, self.X[idx])
            X.append(self.transform(Image.open(file_path)))
            y.append(self.y[idx])
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class DataGeneratorTest(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset):
        self.X, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return int(np.floor(len(self.X)))

    def __getitem__(self, index):
        X = self.__data_generation(index)
        X = np.stack(X)
        return np.moveaxis(X, 1, 3)

    def __data_generation(self, index):
        X = []
        file_path = osp.join(self.image_dir, self.X[index])
        X.append(self.transform(Image.open(file_path)))
        return X


class DataGeneratorCompat(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = params['n_classes']
        self.X, self.y, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X_1, X_2, y = self.__data_generation(indexes)
        X_1, X_2, y = np.stack(X_1), np.stack(X_2), np.stack(y)
        X_3 = tensorflow.concat([np.moveaxis(X_1, 1, 3), np.moveaxis(X_2, 1, 3)], axis=3)
        return X_3, np.array(y)

    def __data_generation(self, indexes):
        X_1 = []; X_2 = []; y = []
        for idx in indexes:
            file_path_1 = osp.join(self.image_dir, self.X[idx][0])
            file_path_2 = osp.join(self.image_dir, self.X[idx][1])
            X_1.append(self.transform(Image.open(file_path_1)))
            X_2.append(self.transform(Image.open(file_path_2)))
            y.append(self.y[idx])
        return X_1, X_2, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class DataGeneratorSiamese(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.n_classes = params['n_classes']
        self.X, self.y, self.transform = dataset
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X_1, X_2, y = self.__data_generation(indexes)
        X_1, X_2, y = np.stack(X_1), np.stack(X_2), np.stack(y)
        X_1 = np.moveaxis(X_1, 1, 3)
        X_2 = np.moveaxis(X_2, 1, 3)
        return [X_1, X_2], np.array(y)

    def __data_generation(self, indexes):
        X_1 = []; X_2 = []; y = []
        for idx in indexes:
            file_path_1 = osp.join(self.image_dir, self.X[idx][0])
            file_path_2 = osp.join(self.image_dir, self.X[idx][1])
            X_1.append(self.transform(Image.open(file_path_1)))
            X_2.append(self.transform(Image.open(file_path_2)))
            y.append(self.y[idx])
        return X_1, X_2 , y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
