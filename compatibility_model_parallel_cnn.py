"""
@shilpath
Objective: Compatibility prediction of two outfits, ie, given two outfits, predict if they are compatible or not.
This file contains a custom CNN model based on parallel CNN architecture trained on the Polyvore outfits dataset.
The architecure of the model is available in the 'compatibility_parallel_cnn_architecture.png' file.
"""
import data_prep
from data import polyvore_dataset, DataGeneratorCompat
from utils import Config
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Lambda, Concatenate, concatenate
from tensorflow.keras import datasets, layers, models, optimizers
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import h5py
import numpy as np

if __name__ == '__main__':

    train_pairwise_file, valid_pairwise_file = data_prep.prepare_data()
    train_pairwise_file = train_pairwise_file
    valid_pairwise_file = valid_pairwise_file
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, y_train = dataset.create_compatibility_dataset2(train_pairwise_file)
    X_test, y_test = dataset.create_compatibility_dataset2(valid_pairwise_file)

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    n_classes = 2
    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator = DataGeneratorCompat(train_set, dataset_size, params)
    test_generator = DataGeneratorCompat(test_set, dataset_size, params)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input = tf.keras.Input(shape=(224, 224, 6))
        split_1 = Lambda(lambda x: x[:, :, :, :3])(input)
        input_1 = Conv2D(32, (3,3), activation='relu')(split_1)
        input_1 = MaxPooling2D(pool_size=(3,3))(input_1)
        input_1 = Conv2D(56, (3,3), activation='relu')(input_1)
        input_1 = MaxPooling2D(pool_size=(3,3))(input_1)
        input_1 = Flatten()(input_1)

        split_2 = Lambda(lambda x: x[:, :, :, 3:6])(input)
        input_2 = Conv2D(32, (3,3), activation='relu')(split_2)
        input_2 = MaxPooling2D(pool_size=(3,3))(input_2)
        input_2 = Conv2D(56, (3,3), activation='relu')(input_2)
        input_2 = MaxPooling2D(pool_size=(3,3))(input_2)
        input_2 = Flatten()(input_2)

        input_3 = concatenate([input_1, input_2])

        input_4 = layers.Dense(128, activation='relu')(input_3)
        outputs = layers.Dense(1, activation='sigmoid')(input_4)

        model = Model(inputs=input, outputs=outputs)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='compatibility_parallel_cnn_architecture.png', show_shapes=True, show_layer_names=True)
    # training
    results = model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        use_multiprocessing=False,
                        workers=Config['num_workers'],
                        epochs=Config['num_epochs']
                        )

    model.save('compat_parallel_cnn_model.hdf5')

    loss = results.history['loss']
    val_loss = results.history['val_loss']
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']

    np.savetxt('acc_compat_cnn.txt', acc)
    np.savetxt('val_acc_compat_cnn.txt', val_acc)

    print("code finished ..")

