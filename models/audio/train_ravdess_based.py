import os
import sys
sys.path.append("../../utils/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, ELU, ReLU
from keras.layers import Convolution2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.applications import InceptionV3
from keras.optimizers import Adam
from keras.regularizers import L2
from keras.callbacks import ModelCheckpoint
from train_network import train_val_test_split
from draw_plot import plot_acc_loss


def define_ravdess_based_model(input_shape, nb_classes):
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(256, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(512, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(1024, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model

if __name__ == "__main__":
    path = "../../database/melgrams/gray/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray"
    files_nb = 1900
    IMG_HEIGHT = 128
    IMG_WIDTH = 1292
    NUM_CLASSES = 4
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    L2_LAMBDA = 0.001
    LEARNING_RATE = 1e-5
    CHANNELS = 1

    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss = 'sparse_categorical_crossentropy'
    metrics = ['sparse_categorical_accuracy']
    checkpoint_filepath = "./tmp/checkpoint"
    checkpoint = ModelCheckpoint(checkpoint_filepath,
                                 save_weights_only=True,
                                 monitor='val_sparse_categorical_accuracy',
                                  mode='max',
                                 verbose=0,
                                 save_best_only=True)
    callbacks_list = [checkpoint]
    
    model = define_ravdess_based_model((IMG_WIDTH, IMG_HEIGHT, CHANNELS), 4)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))
    print(model.summary())
    
    train, val, _ = train_val_test_split(path, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT))

    history = model.fit(train,
                        epochs=NUM_EPOCHS,
                        validation_data=val,
                        callbacks=callbacks_list)
    best_accuracy = max(history.history['val_sparse_categorical_accuracy'])
    plot_acc_loss(history, f"./histories/new_history_ravdess_gray_{BATCH_SIZE}_{LEARNING_RATE}_{best_accuracy}")
    
    model.load_weights(checkpoint_filepath)
    model_path = f"./trained_models/new_ravdess_gray_{BATCH_SIZE}_{LEARNING_RATE}_{best_accuracy}.tf"
    model.save(model_path, overwrite=True, save_format="tf")
