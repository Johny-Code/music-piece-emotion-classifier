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


def define_sarkar_VGG_customized_architecture(input_shape, nb_classes, lambda_value):
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
    model.add(Dense(256, activation="relu", kernel_regularizer=L2(lambda_value)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu", kernel_regularizer=L2(lambda_value)))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
    return model


def train_network(path, batch_size, l2_lambda, learning_rate, epochs, img_height, img_width):
    files_nb = 1900
    NUM_CLASSES = 4
    CHANNELS = 1
    
    optimizer = Adam(learning_rate=learning_rate)
    loss = 'sparse_categorical_crossentropy'
    metrics = ['sparse_categorical_accuracy']
    checkpoint_filepath = "./tmp/checkpoint2"
    checkpoint = ModelCheckpoint(checkpoint_filepath,
                                 save_weights_only=True,
                                 monitor='val_sparse_categorical_accuracy',
                                 mode='max',
                                 verbose=0,
                                 save_best_only=True)
    callbacks_list = [checkpoint]

    model = define_sarkar_VGG_customized_architecture((img_width, img_height, CHANNELS), NUM_CLASSES, l2_lambda)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=(None, img_width, img_height, CHANNELS))
    print(model.summary())
    
    train, val, _ = train_val_test_split(path, batch_size, (img_width, img_height))

    history = model.fit(train,
                        epochs=epochs,
                        validation_data=val,
                        callbacks=callbacks_list)
    best_accuracy = max(history.history['val_sparse_categorical_accuracy'])
    plot_acc_loss(history, f"./histories/different-params/new_history_sarkar_{path[-50:]}_{best_accuracy}")
    
    model.load_weights(checkpoint_filepath)
    model_path = f"./trained_models/different-params/new_sarkar_gray_{path[-50:]}_{best_accuracy}.tf"
    model.save(model_path, overwrite=True, save_format="tf")


if __name__ == "__main__":
    # path = "../../database/melgrams/gray/different-params/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray" 
    NUM_EPOCHS = 700
    BATCH_SIZE = 16
    L2_LAMBDA = 1e-3
    LEARNING_RATE = 1e-5
    IM_WIDTH = 1292
    IM_HEIGHT = 128
    
    paths = ["../../database/melgrams/gray/different-params/melgrams_1024_nfft_256_hop_128_mel_jpg_proper_gray",
             "../../database/melgrams/gray/different-params/melgrams_1024_nfft_256_hop_96_mel_jpg_proper_gray",
             "../../database/melgrams/gray/different-params/melgrams_512_nfft_256_hop_128_mel_jpg_proper_gray",
             "../../database/melgrams/gray/different-params/melgrams_512_nfft_256_hop_96_mel_jpg_proper_gray"]
    img_widths = [5168, 5168, 5168, 5168]
    img_heights = [128, 96, 128, 96]
    
    for i,path in enumerate(paths):
        train_network(path=path, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, l2_lambda=L2_LAMBDA, 
                      epochs=NUM_EPOCHS, img_width=img_widths[i], img_height=img_heights[i])
    