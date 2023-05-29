import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def draw_confusion_matrix(cm, target_names, cmap=None, output_path=None):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if output_path:
        actual_date = datetime.datetime.now()
        output_path_name = os.path.join(output_path,
                                        f'{actual_date.year}-{actual_date.month}-{actual_date.day}_{actual_date.hour}-{actual_date.minute}-{actual_date.second}_cm.png')
        plt.savefig(f"{output_path}_cm.png")
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
