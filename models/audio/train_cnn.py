from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, ELU, ReLU
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend
from keras.utils import np_utils
from os.path import isfile


def get_class_names(path="Samples/"):  # class names are subdirectory names in Preproc/ directory
    class_names = os.listdir(path)
    return class_names


def get_total_files(path="Samples/", train_percentage=0.8): 
    sum_total = 0
    sum_train = 0
    sum_test = 0
    subdirs = os.listdir(path)
    for subdir in subdirs:
        files = os.listdir(path+subdir)
        n_files = len(files)
        sum_total += n_files
        n_train = int(train_percentage*n_files)
        n_test = n_files - n_train
        sum_train += n_train
        sum_test += n_test
    return sum_total, sum_train, sum_test


def get_sample_dimensions(path='Samples/'):
    classname = os.listdir(path)[0]
    files = os.listdir(path+classname)
    infilename = files[0]
    audio_path = path + classname + '/' + infilename
    melgram = np.load(audio_path)
    print("   get_sample_dimensions: melgram.shape = ",melgram.shape)
    return melgram.shape
 

def encode_class(class_name, class_names):  # makes a "one-hot" vector for each class name called
    try:
        idx = class_names.index(class_name)
        vec = np.zeros(len(class_names))
        vec[idx] = 1
        return vec
    except ValueError:
        return None


def shuffle_XY_paths(X,Y,paths):   # generates a randomized order, keeping X&Y(&paths) together
    assert (X.shape[0] == Y.shape[0] )
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths
    for i in range(len(idx)):
        newX[i] = X[idx[i],:,:]
        newY[i] = Y[idx[i],:]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths


def define_oryg_model(nb_layers, kernel_size, input_shape, pool_size, nb_classes):
    nb_filters = 64
    
    model = Sequential()
    model.add(Convolution2D(64, kernel_size[0], kernel_size[1], padding='same', input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    for layer in range(nb_layers-1):
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


def define_128_256_oryg_model(nb_layers, kernel_size, input_shape, pool_size, nb_classes):
    model = Sequential()
    model.add(Convolution2D(64, kernel_size[0], kernel_size[1], padding='same', input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    for layer in range(2):
        model.add(Convolution2D(128, kernel_size[0], kernel_size[1], padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        model.add(Dropout(0.25))
        
    for layer in range(2):
        model.add(Convolution2D(256, kernel_size[0], kernel_size[1], padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(ELU(alpha=1.0))
        model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


def define_Szymons_CNN_model(nb_classes, input_shape):
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3,3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    model.add(Convolution2D(128, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    model.add(Convolution2D(256, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    model.add(Convolution2D(512, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    
    model.add(Convolution2D(1024, kernel_size=(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


def define_panotti_model(kernel_size, input_shape, pool_size, nb_classes, nb_layers=4):
    nb_filters = 32
    cl_dropout = 0.5
    dl_dropout = 0.6
    
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size, padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_size, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))

    for layer in range(nb_layers-1):
        model.add(Convolution2D(nb_filters, kernel_size, padding='same'))
        model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        model.add(Activation("elu"))
        model.add(Dropout(cl_dropout))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('elu'))
    model.add(Dropout(dl_dropout))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


def build_datasets(train_percentage=0.8, preproc=False, path="Samples/"):
    class_names = get_class_names(path=path)
    print("class_names = ",class_names)

    total_files, total_train, total_test = get_total_files(path=path, train_percentage=train_percentage)
    print("total files = ",total_files)

    nb_classes = len(class_names)

    # pre-allocate memory for speed (old method used np.concatenate, slow)
    mel_dims = get_sample_dimensions(path=path)  # Find out the 'shape' of each data file
    X_train = np.zeros((total_train, mel_dims[1], mel_dims[2], mel_dims[3]))   
    Y_train = np.zeros((total_train, nb_classes))  
    X_test = np.zeros((total_test, mel_dims[1], mel_dims[2], mel_dims[3]))  
    Y_test = np.zeros((total_test, nb_classes))  
    paths_train = []
    paths_test = []
    sr = 44100

    train_count = 0
    test_count = 0
    for idx, classname in enumerate(class_names):
        this_Y = np.array(encode_class(classname,class_names) )
        this_Y = this_Y[np.newaxis,:]
        class_files = os.listdir(path+classname)
        n_files = len(class_files)
        n_load =  n_files
        n_train = int(train_percentage * n_load)
        printevery = 100
        print("")
        for idx2, infilename in enumerate(class_files[0:n_load]):          
            audio_path = path + classname + '/' + infilename
            if (0 == idx2 % printevery):
                print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(classname,idx+1,nb_classes),
                       ", file ",idx2+1," of ",n_load,": ",audio_path,sep="")

            melgram = np.load(audio_path)

            if (idx2 < n_train):
                X_train[train_count,:,:] = melgram
                Y_train[train_count,:] = this_Y
                paths_train.append(audio_path)
                train_count += 1
            else:
                X_test[test_count,:,:] = melgram
                Y_test[test_count,:] = this_Y
                paths_test.append(audio_path)
                test_count += 1
        print("")

    print("Shuffling order of data...")
    X_train, Y_train, paths_train = shuffle_XY_paths(X_train, Y_train, paths_train)
    X_test, Y_test, paths_test = shuffle_XY_paths(X_test, Y_test, paths_test)

    return X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr



def build_model(X, Y, nb_classes):
    nb_filters = 128
    pool_size = (2, 2)
    kernel_size = (3, 3)
    nb_layers = 4
    input_shape = (1, X.shape[2], X.shape[3])

    #return define_oryg_model(nb_layers, kernel_size, input_shape, pool_size, nb_classes)
    #return define_128_256_oryg_model(nb_layers, kernel_size, input_shape, pool_size, nb_classes)
    # return define_Szymons_CNN_model(nb_classes, input_shape)
    return define_panotti_model(kernel_size, input_shape, pool_size, nb_classes)


if __name__ == '__main__':
    path = "../../database/melgrams/melgrams_2048_nfft_512_hop_orig/"
    
    np.random.seed(1)
    X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr = build_datasets(preproc=True, path=path)
    
    model = build_model(X_train, Y_train, nb_classes=len(class_names))
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta', #adam is not accurate in this case
              metrics=['accuracy'])
    model.summary()

    load_checkpoint = True
    checkpoint_filepath = 'weights.hdf5'
    if (load_checkpoint):
        print("Looking for previous weights...")
        if ( isfile(checkpoint_filepath) ):
            print ('Checkpoint file detected. Loading weights.')
            model.load_weights(checkpoint_filepath)
        else:
            print ('No checkpoint file detected.  Starting from scratch.')
    else:
        print('Starting from scratch (no checkpoint)')
    checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True)

    batch_size = 32
    nb_epoch = 200
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
