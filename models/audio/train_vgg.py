import os
import sys
sys.path.append("../../utils/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, ELU, ReLU
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.regularizers import L2
from keras.callbacks import ModelCheckpoint
from keras import models
from train_network import train_val_test_split
from draw_plot import plot_acc_loss


def define_VGG_model(input_shape, nb_classes):
    conv_base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    print("input shape", input_shape)
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(512, name='dense_1', kernel_regularizer=L2(0.001)))
    model.add(Dropout(rate=0.3, name='dropout_1'))
    model.add(Activation(activation='relu', name='activation_1'))
    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
    conv_base.trainable = False
    return model


if __name__ == "__main__":
    path = "../../database/melgrams/rgb/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper"
    best_model_path = "vgg_63.19acc_transfer_learning.h5"
    IMG_HEIGHT = 128
    IMG_WIDTH = 1292
    NUM_CLASSES = 4
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    CHANNELS = 3

    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss = 'sparse_categorical_crossentropy'
    metrics = ['sparse_categorical_accuracy']
    checkpoint_filepath = "./tmp/checkpoint3"
    checkpoint = ModelCheckpoint(checkpoint_filepath,
                                 save_weights_only=True,
                                 monitor='val_sparse_categorical_accuracy',
                                 mode='max',
                                 verbose=0,
                                 save_best_only=True)
    callbacks_list = [checkpoint]


    model = define_VGG_model((IMG_WIDTH, IMG_HEIGHT, CHANNELS), NUM_CLASSES)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=(None, IMG_WIDTH, IMG_HEIGHT, CHANNELS))
    print(model.summary())
    
    train, val, _ = train_val_test_split(path, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT), False)

    history = model.fit(train,
                        epochs=NUM_EPOCHS,
                        validation_data=val,
                        callbacks=callbacks_list)
    plot_acc_loss(history, "./histories/new_history_vgg16_regular_resized")
    
    model.load_weights(checkpoint_filepath)
    model_path = "./trained_models/new_vgg16_regular_resized.tf"
    model.save(model_path, overwrite=True, save_format="tf")
    