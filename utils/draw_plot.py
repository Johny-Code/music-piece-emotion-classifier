import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def draw_confusion_matrix(cm, target_names, output_path = None, filename_prefix="", cmap = None):
    os.makedirs(output_path, exist_ok=True)

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
        output_path_name = os.path.join(output_path, f'{get_current_date_string()}')
        plt.savefig(f"{output_path_name}_{filename_prefix}_cm.png")
    else:
        plt.show()
        

def plot_acc_loss_torch(accuracy, loss, path):
    plt.plot(accuracy)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Loss Graph')
    plt.savefig(path+"accuracy.png")
    
    plt.clf()
    plt.plot(loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Accuracy Graph')
    plt.savefig(path+"loss.png")


def plot_acc_loss(history, output_path=None):
    os.makedirs(output_path, exist_ok=True)
    plot_tf_accuracy(history, output_path)
    plot_tf_loss(history, output_path)


def get_current_date_string():
    actual_date = datetime.datetime.now()
    return f"{actual_date.year}-{actual_date.month}-{actual_date.day}_{actual_date.hour}-{actual_date.minute}-{actual_date.second}"


def plot_tf_accuracy(history, output_path):
    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
    except KeyError:
        acc = history.history['sparse_categorical_accuracy']
        val_acc = history.history['val_sparse_categorical_accuracy']

    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if output_path:
        output_path_name = os.path.join(output_path, f'{get_current_date_string()}_acc.png')
        plt.savefig(output_path_name)
    else:
        plt.show()
    
    plt.clf()


def plot_tf_loss(history, output_path):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if output_path:
        output_path_name = os.path.join(output_path,f'{get_current_date_string()}_loss.png')
        plt.savefig(output_path_name)
    else:
        plt.show()
        
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cm', action='store_true')

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


    else:
        print("Use --cm to draw confusion matrix")