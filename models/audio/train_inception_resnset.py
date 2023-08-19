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
    path = "../../database/melgrams/melgrams_2048_nfft_1024_hop_128_mel_jpg_divided_resized/"
    files_nb = 1990
    IMG_HEIGHT = 128
    IMG_WIDTH = 216
    NUM_CLASSES = 4
    NUM_EPOCHS = 60
    BATCH_SIZE = 32
    L2_LAMBDA = 0.001
    LEARNING_RATE = 0.001

    optimizer = Adam(learning_rate=LEARNING_RATE)
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

    model = define_inception_resnet_customized_architecture((IMG_WIDTH, IMG_HEIGHT, 3), 4)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=(None, IMG_WIDTH, IMG_HEIGHT, 3))
    print(model.summary())

    train, val, _ = train_val_test_split(path, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT))

    history = model.fit(train,
                        epochs=NUM_EPOCHS,
                        validation_data=val,
                        callbacks=callbacks_list)
    plot_acc_loss(history, "./histories/new_history_inception_resnet_resized")
    
    model.load_weights(checkpoint_filepath)
    model_path = "./trained_models/new_resnet_inception_resized.tf"
    model.save(model_path, overwrite=True, save_format="tf")
