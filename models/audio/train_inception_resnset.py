import os
import sys
sys.path.append("../../utils/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from train_network import train_val_test_split
from draw_plot import plot_acc_loss
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import L2
from keras.optimizers import Adam
from keras.applications import ResNet50, ResNet152V2
from keras.layers import Conv2D, Flatten, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, ELU, ReLU
from keras.models import Sequential, Model
from implementation.InceptionDepthwise import InceptionDepthwise
from implementation.InceptionResnet import InceptionResnet


def define_inception_resnet_customized_architecture(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 1), padding="same"))

    model.add(InceptionResnet())

    model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(InceptionResnet())
    model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", activation='relu'))

    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())

    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
    return model


def train_network(path, batch_size, learning_rate, img_width, img_height, epochs):
    L2_LAMBDA = 0.001
    NUM_CLASSES = 4
    CHANNELS=1
    
    optimizer = Adam(learning_rate=learning_rate)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    loss = 'sparse_categorical_crossentropy'
    metrics = ['sparse_categorical_accuracy']
    checkpoint_filepath = "./tmp/checkpoint"
    checkpoint = ModelCheckpoint(checkpoint_filepath,
                                 save_weights_only=True,
                                 monitor='val_sparse_categorical_accuracy',
                                 mode='max',
                                 verbose=0,
                                 save_best_only=True)
    callbacks_list = [reduce_lr_callback, checkpoint]

    model = define_inception_resnet_customized_architecture((img_width, img_height, CHANNELS), 4)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # model.build(input_shape=(None, img_width, img_height, CHANNELS))
    print(model.summary())

    train, val, _ = train_val_test_split(path, batch_size, (img_width, img_height))

    history = model.fit(train,
                        epochs=epochs,
                        validation_data=val,
                        callbacks=callbacks_list)
    best_accuracy = max(history.history['val_sparse_categorical_accuracy'])
    plot_acc_loss(history, f"./histories/new_history_inception_resnet_resized1024_bicubic_{batch_size}_{learning_rate}_{best_accuracy}")
    
    model.load_weights(checkpoint_filepath)
    model_path = f"./trained_models/new_resnet_inception_resized1024_bicubic_{batch_size}_{learning_rate}_{best_accuracy}.tf"
    model.save(model_path, overwrite=True, save_format="tf")


if __name__ == "__main__":
    path = "../../database/melgrams/gray/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_1024_width_bicubic/"
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    IMG_HEIGHT = 101
    IMG_WIDTH = 1024
    NUM_EPOCHS = 50
    
    train_network(path, BATCH_SIZE, LEARNING_RATE, IMG_WIDTH, IMG_HEIGHT, NUM_EPOCHS)
