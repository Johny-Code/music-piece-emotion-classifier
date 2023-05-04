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
from train_network import train_val_split, plot_acc_loss


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


def define_fine_tuned_VGG_model(input_shape, nb_classes, model_path):
    conv_base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(512, name='dense_1', kernel_regularizer=L2(0.001)))
    model.add(Dropout(rate=0.3, name='dropout_1'))
    model.add(Activation(activation='relu', name='activation_1'))
    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
    conv_base.trainable = True
    tl_model = models.load_model(filepath=model_path)  # 02 is epoch 3
    tl_model.weights[-4:]  # The weights from the 2 dense layers have to be transferred to the new model
    for i in range(1, len(model.layers)):  # The first layer (index = 0) is the conv base
        model.layers[i].set_weights(tl_model.layers[i].get_weights())
    return model


if __name__ == "__main__":
    path = "../../database/melgrams/melgrams_2048_nfft_512_hop_jpg/"
    best_model_path = "vgg_63.19acc_transfer_learning.h5"
    files_nb = 2000
    IMG_HEIGHT = 216
    IMG_WIDTH = 216
    NUM_CLASSES = 4
    NUM_EPOCHS = 10
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

    # model = define_VGG_model((IMG_WIDTH, IMG_HEIGHT, 3), 4)
    model = define_fine_tuned_VGG_model((IMG_WIDTH, IMG_HEIGHT, 3), 4, best_model_path)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    train, test = train_val_split(path, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT), 0.2)

    history = model.fit(train,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=test,
                        validation_steps=VAL_STEPS,
                        callbacks=[checkpoint])
    plot_acc_loss(history)
