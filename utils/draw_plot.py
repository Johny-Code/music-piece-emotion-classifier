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
        actual_date = datetime.datetime.now()
        output_path_name = os.path.join(output_path,
                                        f'{actual_date.year}-{actual_date.month}-{actual_date.day}_{actual_date.hour}-{actual_date.minute}-{actual_date.second}_acc.png')
        plt.savefig(output_path_name)
    else:
        plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cm', action='store_true')

    args = parser.parse_args()

    if args.cm:

        #feature-based svm best approach
        # cm =   [[41,  7,  8, 15],
        #         [10, 44,13,  4],
        #         [ 5,  9, 35, 18],
        #         [12,  6, 21, 34]]
        
        # output_path = os.path.join('models', 'lyric', 'feature_based', 'svm')
        
        #feature-based ann best approach
        # cm = [[43, 11,  4, 13],
        #       [10, 46, 10,  5],
        #       [ 5, 13, 21, 28],
        #      [15, 15,  7, 36]]

        # output_path = os.path.join('models', 'lyric', 'feature_based', 'ann')

        #fasttext best params
        cm = [[49, 10,  7,  5],
                [11, 29, 10, 21],
                [11,  4, 32, 20],
                [ 5,  8, 29, 31]]

        output_path = os.path.join('models', 'lyric', 'fasttext')

        TARGET_NAMES = ['happy', 'angry', 'sad', 'relaxed']

        draw_confusion_matrix(cm, TARGET_NAMES, output_path, cmap='Blues')


    else:
        print("Use --cm to draw confusion matrix")