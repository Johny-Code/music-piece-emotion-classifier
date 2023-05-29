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
from utils.train_network import train_val_split
from utils.draw_plot import plot_acc_loss


def define_InceptionV3_model(input_shape, nb_classes):
    conv_base = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    print("input shape", input_shape)
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(512, name='dense_1', kernel_regularizer=L2(0.001)))
    model.add(Dropout(rate=0.5, name='dropout_1'))
    model.add(Dense(256, name='dense_2', kernel_regularizer=L2(0.001)))
    model.add(Dropout(rate=0.3, name='dropout_2'))
    model.add(Activation(activation='relu', name='activation_1'))
    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
    conv_base.trainable = False
    return model


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    path = "../../database/melgrams/melgrams_2048_nfft_512_hop_jpg/"
    files_nb = 2000
    IMG_HEIGHT = 216
    IMG_WIDTH = 216
    NUM_CLASSES = 4
    NUM_EPOCHS = 2
    BATCH_SIZE = 32
    L2_LAMBDA = 0.001
    STEPS_PER_EPOCH = int(files_nb * 0.8) // BATCH_SIZE
    VAL_STEPS = int(files_nb * 0.2) // BATCH_SIZE

    optimizer = Adam(lr=1e-5)
    loss = 'sparse_categorical_crossentropy'
    metrics = ['sparse_categorical_accuracy']
    filepath = "./transfer_learning_epoch_{epoch:02d}_{sparse_categorical_accuracy:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_sparse_categorical_accuracy',
                                 verbose=0,
                                 save_best_only=False)
    callbacks_list = [checkpoint]

    model = define_InceptionV3_model((IMG_WIDTH, IMG_HEIGHT, 3), 4)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    train, test = train_val_split(path, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT), 0.2)

    history = model.fit(train,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=test,
                        validation_steps=VAL_STEPS,
                        callbacks=[checkpoint])
    plot_acc_loss (history, "./history")
