import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def draw_confusion_matrix(cm, target_names, output_path = None, cmap = None):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, 
                annot=True,
                cbar=True, 
                fmt='d', 
                cmap=cmap, 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if output_path:
        actual_date = datetime.datetime.now()
        output_path_name = os.path.join(output_path,
                                        f'{actual_date.year}-{actual_date.month}-{actual_date.day}_{actual_date.hour}-{actual_date.minute}-{actual_date.second}_cm.png')
        plt.savefig(f"{output_path_name}_cm.png")
    else:
        plt.show()


def plot_acc_loss(history, output_path=None):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #  "Accuracy"
    try:
        acc = history['accuracy']
        val_acc = history['val_accuracy']
    except KeyError:
        acc = history['sparse_categorical_accuracy']
        val_acc = history['val_sparse_categorical_accuracy']

    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if output_path:
        actual_date = datetime.datetime.now()
        output_path_name = os.path.join(output_path,
                                        f'{actual_date.year}-{actual_date.month}-{actual_date.day}_{actual_date.hour}-{actual_date.minute}-{actual_date.second}_acc.png')
        plt.savefig(output_path_name)
    else:
        plt.show()

    # "Loss"
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if output_path:
        actual_date = datetime.datetime.now()
        output_path_name = os.path.join(output_path,
                                        f'{actual_date.year}-{actual_date.month}-{actual_date.day}_{actual_date.hour}-{actual_date.minute}-{actual_date.second}_loss.png')
        plt.savefig(output_path_name)
    else:
        plt.show()

def plot_acc_loss_v1(history, output_path=None):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #  "Accuracy"
    try:
        acc = history['accuracy']
        val_acc = history['val_accuracy']
    except KeyError:
        acc = history['sparse_categorical_accuracy']
        val_acc = history['val_sparse_categorical_accuracy']

    plt.figure(figsize=(6, 4))
    plt.plot(acc)
    plt.plot(val_acc)
    # plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if output_path:
        actual_date = datetime.datetime.now()
        output_path_name = os.path.join(output_path,
                                        f'{actual_date.year}-{actual_date.month}-{actual_date.day}_{actual_date.hour}-{actual_date.minute}-{actual_date.second}_acc.png')
        plt.savefig(output_path_name)
    else:
        plt.show()

    #clear plt 
    plt.clf()

    # "Loss"
    plt.figure(figsize=(, 4))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    # plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if output_path:
        actual_date = datetime.datetime.now()
        output_path_name = os.path.join(output_path,
                                        f'{actual_date.year}-{actual_date.month}-{actual_date.day}_{actual_date.hour}-{actual_date.minute}-{actual_date.second}_loss.png')
        plt.savefig(output_path_name)
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cm', action='store_true')
    parser.add_argument('--loss_acc', action='store_true')

    args = parser.parse_args()

    if args.cm:

        #feature-based svm best approach
        cm =   [[41,  7,  8, 15],
                [10, 44,13,  4],
                [ 5,  9, 35, 18],
                [12,  6, 21, 34]]
        
        output_path = os.path.join('models', 'lyric', 'feature_based', 'svm')
        
        #feature-based ann best approach
        # cm = [[48,  4,  6, 13],
        #       [11, 42, 17,  1],
        #       [ 5,  5, 32, 25],
        #       [15,  4, 18, 36]]

        # output_path = os.path.join('models', 'lyric', 'feature_based', 'ann')

        #fasttext-based autotune 30 minut 
        # cm = [[45, 15,  2,  9],
        #       [15, 39, 12,  5],
        #       [ 6, 14, 31, 22],
        #       [14,  6, 18, 29]]

        # output_path = os.path.join('models', 'lyric', 'fasttext')

        TARGET_NAMES = ['happy', 'angry', 'sad', 'relaxed']

        draw_confusion_matrix(cm, TARGET_NAMES, output_path, cmap='Blues')

    if args.loss_acc:

        history ={
            "loss" : [2.1655, 1.3698, 1.2839, 1.2314, 1.1725, 1.1216, 1.0629, 1.0569, 1.0136, 0.9408, 0.9218, 0.9043, 0.8927, 0.8427, 0.7998],
            "accuracy": [0.2777, 0.3323, 0.4036, 0.4537, 0.4628, 0.5106, 0.5417, 0.5296, 0.5766, 0.5986, 0.6055, 0.6222, 0.6146, 0.6495, 0.6753],
            "val_loss": [1.3535, 1.2882, 1.2094, 1.2167, 1.1475, 1.1214, 1.1019, 1.1696, 1.1272, 1.1307, 1.1523, 1.1588, 1.1201, 1.1548, 1.1684],
            "val_accuracy": [0.3322, 0.3640, 0.4417, 0.4028, 0.4629, 0.4947, 0.4982, 0.4664, 0.5124, 0.4700, 0.5265, 0.4806, 0.5230, 0.5194, 0.5300]
        }
    

        output_path = os.path.join('models', 'lyric', 'feature_based', 'ann')

        plot_acc_loss_v1(history, output_path)

    else:
        print("Use --cm to draw confusion matrix")