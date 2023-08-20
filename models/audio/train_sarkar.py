import os
import sys
sys.path.append("../../utils/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from train_network import train_val_test_split
from draw_plot import plot_acc_loss
from keras.callbacks import ModelCheckpoint
from keras.regularizers import L2
from keras.optimizers import Adam
from keras.applications import ResNet50, ResNet152V2
from keras.layers import Conv2D, Flatten, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, ELU, ReLU
from keras.models import Sequential, Model


def define_sarkar_VGG_customized_architecture(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same", input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding="same"))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu", kernel_regularizer=L2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu", kernel_regularizer=L2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
    return model


if __name__ == "__main__":
    path = "../../database/melgrams/gray/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray"
    files_nb = 1900
    IMG_HEIGHT = 128
    IMG_WIDTH = 1292
    NUM_CLASSES = 4
    NUM_EPOCHS = 700
    BATCH_SIZE = 16
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

    model = define_sarkar_VGG_customized_architecture((IMG_WIDTH, IMG_HEIGHT, CHANNELS), 4)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=(None, IMG_WIDTH, IMG_HEIGHT, CHANNELS))
    print(model.summary())
    
    train, val, _ = train_val_test_split(path, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT))

    history = model.fit(train,
                        epochs=NUM_EPOCHS,
                        validation_data=val,
                        callbacks=callbacks_list)
    best_accuracy = max(history.history['val_sparse_categorical_accuracy'])
    plot_acc_loss(history, f"./histories/new_history_sarkar_gray_{BATCH_SIZE}_{LEARNING_RATE}_{best_accuracy}")
    
    model.load_weights(checkpoint_filepath)
    model_path = f"./trained_models/new_sarkar_gray_{BATCH_SIZE}_{LEARNING_RATE}_{best_accuracy}.tf"
    model.save(model_path, overwrite=True, save_format="tf")

