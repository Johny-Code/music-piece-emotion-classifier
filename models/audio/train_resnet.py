import os
import sys
sys.path.append("../../utils/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from implementation.ResNet18 import ResNet18
from train_network import train_val_test_split 
from draw_plot import plot_acc_loss
from keras.callbacks import ModelCheckpoint
from keras.regularizers import L2
from keras.optimizers import Adam
from keras.applications import ResNet50, ResNet152V2
from keras.layers import Convolution2D, Flatten, GlobalAveragePooling2D
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, ELU, ReLU
from keras.models import Sequential, Model


def define_Resnet18_model(input_shape, nb_classes):
    model = ResNet18(nb_classes)
    model.build(input_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
    return model


#fine-tuning
def define_Resnet50_partial_model(input_shape, nb_classes):
    conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(1024, name='dense_1', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(512, name='dense_2', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, name='dense_3', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))

    for layer in conv_base.layers[0:143]:
        layer.trainable = False
    for layer in conv_base.layers[143:]:
        layer.trainable = True
    return model


#transfer-learning
def define_fine_tuned_Resnet50_full_model(input_shape, nb_classes):
    conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    model = Sequential()
    model.add(conv_base)
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


#transfer-learning
def define_Resnet50_full_model(input_shape, nb_classes):
    conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    model = Sequential()
    model.add(conv_base)
    conv_base.trainable = False
    # model.add(GlobalAveragePooling2D()) #worse performance
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
    return model


#fine-tuning
def define_fine_tuned_Resnet152V2_partial_model(input_shape, nb_classes, non_trainable_layers_nb):
    conv_base = ResNet152V2(include_top=False, weights='imagenet', input_shape=input_shape, pooling="avg")
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))

    for layer in conv_base.layers[0:non_trainable_layers_nb]:
        layer.trainable = False
    for layer in conv_base.layers[non_trainable_layers_nb:]:
        layer.trainable = True
    return model


#transfer-learning
def define_fine_tuned_Resnet152V2_full_model(input_shape, nb_classes):
    conv_base = ResNet152V2(include_top=False, weights='imagenet', input_shape=input_shape)
    model = Sequential()
    model.add(conv_base)
    conv_base.trainable = False
    model.add(Flatten())
    model.add(Dense(2048, name='dense_0', kernel_regularizer=L2(0.001), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax', name='dense_output'))
    return model


if __name__ == "__main__":
    path = "../../database/melgrams/rgb/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper"
    IMG_HEIGHT = 128
    IMG_WIDTH = 1292
    NUM_CLASSES = 4
    NUM_EPOCHS = 7
    BATCH_SIZE = 16
    L2_LAMBDA = 0.001
    LEARNING_RATE = 1e-5
    CHANNELS = 3

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

    # model = define_Resnet18_model((IMG_WIDTH, IMG_HEIGHT, CHANNELS), 4)
    # model = define_Resnet50_full_model((IMG_WIDTH, IMG_HEIGHT, CHANNELS), 4)
    # model = define_Resnet50_partial_model((IMG_WIDTH, IMG_HEIGHT, CHANNELS), 4)
    # model = define_fine_tuned_Resnet152V2_full_model((IMG_WIDTH, IMG_HEIGHT, CHANNELS), 4)
    model = define_fine_tuned_Resnet152V2_partial_model((IMG_WIDTH, IMG_HEIGHT, CHANNELS), 4, 142)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(model.summary())
    
    train, val, _ = train_val_test_split(path, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT), False)

    history = model.fit(train,
                        epochs=NUM_EPOCHS,
                        validation_data=val,
                        callbacks=callbacks_list)
    best_accuracy = max(history.history['val_sparse_categorical_accuracy'])
    plot_acc_loss(history, f"./histories/new_history_resnet152V2_finetuned_resized_{BATCH_SIZE}_{LEARNING_RATE}_{best_accuracy}")
    
    model.load_weights(checkpoint_filepath)
    model_path = f"./trained_models/new_resnet152V2_finetuned_resized_{BATCH_SIZE}_{LEARNING_RATE}_{best_accuracy}.tf"
    model.save(model_path, overwrite=True, save_format="tf")
