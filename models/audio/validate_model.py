import os
import sys
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("../../utils/")

from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import load_img,img_to_array
from sklearn.metrics import confusion_matrix
from draw_plot import draw_confusion_matrix
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


if __name__ == "__main__":
    MODEL_NAME = "./trained_models/new_resnet_inception_resized_59_53.tf"
    IMG_HEIGHT = 128
    IMG_WIDTH = 216
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    path = "../../database/melgrams/melgrams_2048_nfft_1024_hop_128_mel_jpg_divided_resized/train" #happy, relaxed ok; sad, angry sredniawo
    optimizer = Adam(learning_rate=LEARNING_RATE)
    loss = 'sparse_categorical_crossentropy'
    metrics = ['sparse_categorical_accuracy']
    label_mapping = {"angry":0, "happy": 1, "relaxed": 2, "sad": 3}
    emotions_mapping = {"sad": "sad", "happy": "happy", "angry": "angry", "relaxed": "relaxed"}
    
    img_paths = []
    input_images = []
    true_labels = []
    
    for emotion in label_mapping.keys():
        emotion_files = os.listdir(os.path.join(path, emotion))
        for file in emotion_files:
            img_paths.append(os.path.join(path, emotion, file))
    
    for img_path in img_paths:
        label = next((mapped_label for keyword, mapped_label in emotions_mapping.items() if keyword in img_path))
        image = load_img(img_path)
        input_arr = img_to_array(image)
        input_arr = input_arr / 255.0
        input_images.append(input_arr)
        true_labels.append(label)

    input_images = np.array(input_images)
    fixed_input_data = np.transpose(input_images, (0, 2, 1, 3))

    model = load_model(MODEL_NAME)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    predictions = model.predict(fixed_input_data)
    predicted_labels = np.argmax(predictions, axis=1)
    
    true_labels_int = [label_mapping[label] for label in true_labels]
    cfm = confusion_matrix(true_labels_int, predicted_labels)
    draw_confusion_matrix(cfm, label_mapping.keys(), "confusion_matrices", filename_prefix="inception-resnet")
    


    
    