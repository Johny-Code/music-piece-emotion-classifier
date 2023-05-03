from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def preprocess_image(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label


def train_val_split(path, batch_size, img_shape, val_split=0.2):
    train_ds = image_dataset_from_directory(
        path,
        labels="inferred",
        validation_split=val_split,
        seed=123,
        subset='training',
        color_mode='rgb',
        image_size=img_shape,
        batch_size=batch_size)
    
    validation_ds= image_dataset_from_directory(
        path,
        labels="inferred",
        validation_split=val_split,
        seed=123,
        subset='validation',
        color_mode='rgb',
        image_size=img_shape,
        batch_size=batch_size)
    
    return train_ds.map(preprocess_image), validation_ds.map(preprocess_image)


def plot_acc_loss(history, directory="./history"):
    os.makedirs(f"{directory}", exist_ok=True)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig(f"{directory}/LossVal_loss.png")
    plt.clf()
    plt.plot(history.history['sparse_categorical_accuracy'], label='train acc')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='val acc')
    plt.legend()
    plt.savefig(f"{directory}/AccVal_acc.png")
