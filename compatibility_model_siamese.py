"""
@shilpath
Objective: Compatibility prediction of two outfits, ie, given two outfits, predict if they are compatible or not.
This file contains a custom CNN model based on the Siamese architecture trained on the Polyvore outfits dataset.
The architecure of the model is available in the 'compatibility_siamese_model_architecture.png' file.
"""
from data import polyvore_dataset, DataGeneratorSiamese
from utils import Config
import data_prep
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Lambda, Input
from tensorflow.keras import models, optimizers
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
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

    train_generator = DataGeneratorSiamese(train_set, dataset_size, params)
    test_generator = DataGeneratorSiamese(test_set, dataset_size, params)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inp_shape = (224,224,3)
        left_inp = Input(inp_shape)
        right_inp = Input(inp_shape)

        model = models.Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape=inp_shape))
        model.add(MaxPooling2D((3,3)))

        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((3,3)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(MaxPooling2D((3,3)))

        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(MaxPooling2D((3,3)))

        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='sigmoid'))

        interm_left_output = model(left_inp)
        interm_right_output = model(right_inp)

        euclidean_distance_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        euclidean_dist = euclidean_distance_layer([interm_left_output, interm_right_output])

        predicted_output = Dense(1, activation='sigmoid')(euclidean_dist)

        siamese_cnn = Model(inputs = [left_inp, right_inp], outputs = predicted_output)
        siamese_cnn.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy'])

    siamese_cnn.summary()
    plot_model(siamese_cnn, to_file='compatibility_siamese_model_architecture.png', show_shapes=True, show_layer_names=True)

    results = siamese_cnn.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        use_multiprocessing=False,
                        workers=Config['num_workers'],
                        epochs=Config['num_epochs']
                        )

    siamese_cnn.save('siamese_model.hdf5')

    loss = results.history['loss']
    val_loss = results.history['val_loss']
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']

    np.savetxt('acc_compat.txt', acc)
    np.savetxt('val_acc_compat.txt', val_acc)

    print("code finished ..")



