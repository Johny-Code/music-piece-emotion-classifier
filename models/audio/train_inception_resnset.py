import os
import sys
sys.path.append("../../utils/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from train_network import train_val_split, plot_acc_loss
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
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
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


if __name__ == "__main__":
    path = "../../database/melgrams/melgrams_2048_nfft_1024_hop_128_mel_jpg/"
    files_nb = 200
    IMG_HEIGHT = 216
    IMG_WIDTH = 216
    NUM_CLASSES = 4
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    L2_LAMBDA = 0.001
    TRAIN_SPLIT = 0.8
    LEARNING_RATE = 0.01
    STEPS_PER_EPOCH = int(files_nb * TRAIN_SPLIT) // BATCH_SIZE
    VAL_STEPS = int(files_nb * (1 - TRAIN_SPLIT)) // BATCH_SIZE

    optimizer = Adam(learning_rate=LEARNING_RATE)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    loss = 'sparse_categorical_crossentropy'
    metrics = ['sparse_categorical_accuracy']
    filepath = "./transfer_learning_epoch_{epoch:02d}_{sparse_categorical_accuracy:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_sparse_categorical_accuracy',
                                 verbose=0,
                                 save_best_only=False)
    callbacks_list = [reduce_lr_callback]  # , checkpoint]

    model = define_inception_resnet_customized_architecture((IMG_WIDTH, IMG_HEIGHT, 3), 4)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=(None, IMG_WIDTH, IMG_HEIGHT, 3))
    print(model.summary())

    train, test = train_val_split(path, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT), (1 - TRAIN_SPLIT))

    history = model.fit(train,
                        epochs=NUM_EPOCHS,
                        # steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=test,)
                        # validation_steps=VAL_STEPS,)
                        # callbacks=callbacks_list)
    plot_acc_loss(history)
