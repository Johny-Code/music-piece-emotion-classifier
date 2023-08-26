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


def train_val_test_split(path, batch_size, img_shape, use_one_channel=True):
    color = "grayscale" if use_one_channel else "rgb"
    
    train_ds = image_dataset_from_directory(
        os.path.join(path, 'train'),
        labels="inferred",
        color_mode=color,
        image_size=img_shape,
        batch_size=batch_size)
    
    val_ds = image_dataset_from_directory(
        os.path.join(path, 'val'),
        labels="inferred",
        color_mode=color,
        image_size=img_shape,
        batch_size=batch_size)
    
    test_ds = image_dataset_from_directory(
        os.path.join(path, 'test'),
        labels="inferred",
        color_mode=color,
        image_size=img_shape,
        batch_size=batch_size)

    return train_ds.map(preprocess_image), val_ds.map(preprocess_image), test_ds.map(preprocess_image)
