from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, ELU, ReLU
from keras.layers import Convolution2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.utils import image_dataset_from_directory
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.regularizers import L2
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def define_Resnet50_partial_model(input_shape, nb_classes):
    conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, name='dense_1', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, name='dense_2', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, name='dense_3', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
        
    for layer in conv_base.layers[0:143]:
      layer.trainable = False
    for layer in conv_base.layers[143:]:
        layer.trainable = True
    
    return model


def define_Resnet50_full_model(input_shape, nb_classes):
    conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    model = Sequential()
    model.add(conv_base)
    # model.add(GlobalAveragePooling2D())
    # model.add(Flatten())
    # model.add(Dense(1024, name='dense_1', kernel_regularizer=L2(0.001)))
    # model.add(Activation(activation='relu', name='activation_1'))
    # model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
    conv_base.trainable = False
    
    model.add(Flatten())
    model.add(Dense(1024, name='dense_0', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, name='dense_1', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, name='dense_2', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, name='dense_3', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
        
    return model


def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label


def train_val_split(path, batch_size, img_shape):
    train_ds, validation_ds = image_dataset_from_directory(
        path,
        labels="inferred",
        validation_split=0.2,
        seed=123,
        subset='both',
        color_mode='rgb',
        image_size=img_shape,
        batch_size=batch_size)
    return train_ds.map(process), validation_ds.map(process)


def plot_acc_loss(history):
    os.makedirs("./history", exist_ok=True)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig('./history/LossVal_loss.png')
    plt.clf()
    plt.plot(history.history['sparse_categorical_accuracy'], label='train acc')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='val acc')
    plt.legend()
    plt.savefig('./history/AccVal_acc.png')


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    path = "../../database/melgrams/melgrams_2048_nfft_512_hop_jpg/"
    files_nb = 2000
    IMG_HEIGHT = 216
    IMG_WIDTH = 216
    NUM_CLASSES = 4
    NUM_EPOCHS = 1000
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

    #model = define_Resnet50_partial_model((IMG_WIDTH, IMG_HEIGHT, 3), 4)
    model = define_Resnet50_full_model((IMG_WIDTH, IMG_HEIGHT, 3), 4)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    train, test = train_val_split(path, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT))

    history = model.fit(train,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=test,
                        validation_steps=VAL_STEPS,)
                        #callbacks=[checkpoint])
    plot_acc_loss(history)
