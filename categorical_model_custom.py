"""
@shilpath
Objective: Category prediction of outfits.
This file contains a custom CNN model trained on the Polyvore outfits dataset.
"""
from data import polyvore_dataset, DataGenerator
from utils import Config

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
import numpy as np

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator = DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Sequential()

        # BLOCK 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        # BLOCK 2
        model.add(layers.Conv2D(56, (3, 3), activation='relu'))
        model.add(layers.Conv2D(56, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        # BLOCK 3
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # BLOCK 4
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        # FULLY CONNECTED
        model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.Dense(n_classes, activation='softmax'))

        model.compile(
            optimizer=optimizers.Adam(lr=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    model.summary()

    plot_model(model, to_file='my_model.png', show_shapes=True, show_layer_names=True)

    results = model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        use_multiprocessing=True,
                        workers=Config['num_workers'],
                        epochs=Config['num_epochs']
                        )

    model.save('polyvore_trained_mymodel.hdf5')

    loss = results.history['loss']
    val_loss = results.history['val_loss']
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']

    np.savetxt('acc_custom.txt', acc)
    np.savetxt('val_acc_custom.txt', val_acc)

    print("code finished ..") 




