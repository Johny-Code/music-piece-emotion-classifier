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
    plt.figure(figsize=(8, 6))
    plt.plot(acc)
    plt.plot(val_acc)
    plt.ylim(0, 100)
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
    plt.figure(figsize=(8, 6))
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
    parser.add_argument('--loss', action='store_true')

    args = parser.parse_args()

    if args.cm:

        #feature-based svm best approach
        # cm =   [[41,  7,  8, 15],
        #         [10, 44,13,  4],
        #         [ 5,  9, 35, 18],
        #         [12,  6, 21, 34]]
        
        # output_path = os.path.join('models', 'lyric', 'feature_based', 'svm')
        
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


        #transformer-based approach 
        # cm = [[47, 16,  2,  7],
        #       [12, 55,  4,  3],
        #       [ 7, 21, 20, 22],
        #       [ 7,  9,  9, 42]]

        #transformer-based approach full dataset
        # cm = [[49, 16,  2,  8],
        #       [12, 55,  4,  3],
        #       [ 7, 21, 20, 26],
        #       [12,  9,  9, 45]]

        # output_path = os.path.join('models', 'lyric', 'xlnet')
        
        # happy     X     X     X   X
        # angry     X     X     X   X
        # sad       X     X     X   X
        # relaxed   X     X     X   X
        #           happy angry sad relaxed


        #Sarkar et al. 
        # cm = [[61, 8,  0,  6],
        #       [ 29, 39,  3,  3],
        #       [ 10,  11, 20,  33],
        #       [9, 1, 9, 56]]
    
        # output_path = os.path.join('models', 'audio', 'sarkar')

        # Majority voting ensemble
        
        # cm = [[43, 21,  5,  6],
        #         [15, 56,  1,  2],
        #         [ 5, 23, 21, 25],
        #         [ 6,  2,  6, 61]]

        #concatenated

        cm = [[39, 16, 9, 11],
                [18, 44, 11, 1],
                [4,11,37,22],
                [2,1,18,54]]

              


        output_path = os.path.join('models', 'ensemble')

        TARGET_NAMES = ['happy', 'angry', 'sad', 'relaxed']

        draw_confusion_matrix(cm, TARGET_NAMES, output_path, cmap='Blues')

    if args.loss:
        class history:
            def __init__(self, loss, val_loss, accuracy, val_accuracy):
                self.history = {'loss': loss, 
                                'val_loss': val_loss, 
                                'accuracy': accuracy, 
                                'val_accuracy': val_accuracy}
        
# Epoch [1/100] - Train Loss: 1.3831 Val Loss: 1.3825 - Train Acc: 26.70% - Val Acc: 33.22%
# Epoch [2/100] - Train Loss: 1.3795 Val Loss: 1.3785 - Train Acc: 32.38% - Val Acc: 45.64%
# Epoch [3/100] - Train Loss: 1.3750 Val Loss: 1.3741 - Train Acc: 34.67% - Val Acc: 45.64%
# Epoch [4/100] - Train Loss: 1.3680 Val Loss: 1.3691 - Train Acc: 35.82% - Val Acc: 45.64%
# Epoch [5/100] - Train Loss: 1.3628 Val Loss: 1.3635 - Train Acc: 37.33% - Val Acc: 46.31%
# Epoch [6/100] - Train Loss: 1.3574 Val Loss: 1.3572 - Train Acc: 38.12% - Val Acc: 46.31%
# Epoch [7/100] - Train Loss: 1.3510 Val Loss: 1.3504 - Train Acc: 38.12% - Val Acc: 45.97%
# Epoch [8/100] - Train Loss: 1.3431 Val Loss: 1.3432 - Train Acc: 40.63% - Val Acc: 45.64%
# Epoch [9/100] - Train Loss: 1.3321 Val Loss: 1.3352 - Train Acc: 41.42% - Val Acc: 45.64%
# Epoch [10/100] - Train Loss: 1.3245 Val Loss: 1.3271 - Train Acc: 42.93% - Val Acc: 45.30%
# Epoch [11/100] - Train Loss: 1.3136 Val Loss: 1.3186 - Train Acc: 43.29% - Val Acc: 45.30%
# Epoch [12/100] - Train Loss: 1.3058 Val Loss: 1.3101 - Train Acc: 44.58% - Val Acc: 45.64%
# Epoch [13/100] - Train Loss: 1.2989 Val Loss: 1.3022 - Train Acc: 44.87% - Val Acc: 45.30%
# Epoch [14/100] - Train Loss: 1.2888 Val Loss: 1.2944 - Train Acc: 45.66% - Val Acc: 45.97%
# Epoch [15/100] - Train Loss: 1.2815 Val Loss: 1.2870 - Train Acc: 44.36% - Val Acc: 47.32%
# Epoch [16/100] - Train Loss: 1.2721 Val Loss: 1.2800 - Train Acc: 44.94% - Val Acc: 46.98%
# Epoch [17/100] - Train Loss: 1.2637 Val Loss: 1.2731 - Train Acc: 46.88% - Val Acc: 48.99%
# Epoch [18/100] - Train Loss: 1.2548 Val Loss: 1.2667 - Train Acc: 47.95% - Val Acc: 49.33%
# Epoch [19/100] - Train Loss: 1.2511 Val Loss: 1.2606 - Train Acc: 49.39% - Val Acc: 50.34%
# Epoch [20/100] - Train Loss: 1.2404 Val Loss: 1.2545 - Train Acc: 50.75% - Val Acc: 51.01%
# Epoch [21/100] - Train Loss: 1.2319 Val Loss: 1.2491 - Train Acc: 53.05% - Val Acc: 51.34%
# Epoch [22/100] - Train Loss: 1.2265 Val Loss: 1.2438 - Train Acc: 54.27% - Val Acc: 51.34%
# Epoch [23/100] - Train Loss: 1.2167 Val Loss: 1.2383 - Train Acc: 56.07% - Val Acc: 50.67%
# Epoch [24/100] - Train Loss: 1.2084 Val Loss: 1.2332 - Train Acc: 57.14% - Val Acc: 51.68%
# Epoch [25/100] - Train Loss: 1.2000 Val Loss: 1.2285 - Train Acc: 57.50% - Val Acc: 52.35%
# Epoch [26/100] - Train Loss: 1.2004 Val Loss: 1.2243 - Train Acc: 58.72% - Val Acc: 53.69%
# Epoch [27/100] - Train Loss: 1.1915 Val Loss: 1.2204 - Train Acc: 59.58% - Val Acc: 53.69%
# Epoch [28/100] - Train Loss: 1.1843 Val Loss: 1.2167 - Train Acc: 60.52% - Val Acc: 54.36%
# Epoch [29/100] - Train Loss: 1.1821 Val Loss: 1.2136 - Train Acc: 60.37% - Val Acc: 54.36%
# Epoch [30/100] - Train Loss: 1.1758 Val Loss: 1.2108 - Train Acc: 60.09% - Val Acc: 55.03%
# Epoch [31/100] - Train Loss: 1.1696 Val Loss: 1.2082 - Train Acc: 62.17% - Val Acc: 54.70%
# Epoch [32/100] - Train Loss: 1.1664 Val Loss: 1.2056 - Train Acc: 60.88% - Val Acc: 55.03%
# Epoch [33/100] - Train Loss: 1.1576 Val Loss: 1.2035 - Train Acc: 62.60% - Val Acc: 55.37%
# Epoch [34/100] - Train Loss: 1.1561 Val Loss: 1.2017 - Train Acc: 63.32% - Val Acc: 55.70%
# Epoch [35/100] - Train Loss: 1.1559 Val Loss: 1.2000 - Train Acc: 62.89% - Val Acc: 55.37%
# Epoch [36/100] - Train Loss: 1.1477 Val Loss: 1.1981 - Train Acc: 63.68% - Val Acc: 56.38%
# Epoch [37/100] - Train Loss: 1.1480 Val Loss: 1.1965 - Train Acc: 62.67% - Val Acc: 56.38%
# Epoch [38/100] - Train Loss: 1.1437 Val Loss: 1.1950 - Train Acc: 63.96% - Val Acc: 56.71%
# Epoch [39/100] - Train Loss: 1.1427 Val Loss: 1.1934 - Train Acc: 64.03% - Val Acc: 57.05%
# Epoch [40/100] - Train Loss: 1.1379 Val Loss: 1.1923 - Train Acc: 63.82% - Val Acc: 56.71%
# Epoch [41/100] - Train Loss: 1.1348 Val Loss: 1.1907 - Train Acc: 65.04% - Val Acc: 56.04%
# Epoch [42/100] - Train Loss: 1.1349 Val Loss: 1.1897 - Train Acc: 64.11% - Val Acc: 56.38%
# Epoch [43/100] - Train Loss: 1.1317 Val Loss: 1.1888 - Train Acc: 64.47% - Val Acc: 55.37%
# Epoch [44/100] - Train Loss: 1.1302 Val Loss: 1.1880 - Train Acc: 65.18% - Val Acc: 56.04%
# Epoch [45/100] - Train Loss: 1.1277 Val Loss: 1.1871 - Train Acc: 64.18% - Val Acc: 56.04%
# Epoch [46/100] - Train Loss: 1.1309 Val Loss: 1.1859 - Train Acc: 63.68% - Val Acc: 56.04%
# Epoch [47/100] - Train Loss: 1.1261 Val Loss: 1.1852 - Train Acc: 63.53% - Val Acc: 55.70%
# Epoch [48/100] - Train Loss: 1.1264 Val Loss: 1.1846 - Train Acc: 63.53% - Val Acc: 56.04%
# Epoch [49/100] - Train Loss: 1.1238 Val Loss: 1.1842 - Train Acc: 63.53% - Val Acc: 55.70%
# Epoch [50/100] - Train Loss: 1.1149 Val Loss: 1.1841 - Train Acc: 64.03% - Val Acc: 56.71%
# Epoch [51/100] - Train Loss: 1.1195 Val Loss: 1.1833 - Train Acc: 65.04% - Val Acc: 57.05%
# Epoch [52/100] - Train Loss: 1.1149 Val Loss: 1.1821 - Train Acc: 64.54% - Val Acc: 57.05%
# Epoch [53/100] - Train Loss: 1.1204 Val Loss: 1.1817 - Train Acc: 64.90% - Val Acc: 57.05%
# Epoch [54/100] - Train Loss: 1.1104 Val Loss: 1.1810 - Train Acc: 65.25% - Val Acc: 57.05%
# Epoch [55/100] - Train Loss: 1.1172 Val Loss: 1.1803 - Train Acc: 64.47% - Val Acc: 57.05%
# Epoch [56/100] - Train Loss: 1.1169 Val Loss: 1.1799 - Train Acc: 64.39% - Val Acc: 56.71%
# Epoch [57/100] - Train Loss: 1.1143 Val Loss: 1.1797 - Train Acc: 64.75% - Val Acc: 57.05%
# Epoch [58/100] - Train Loss: 1.1121 Val Loss: 1.1790 - Train Acc: 64.39% - Val Acc: 56.71%
# Epoch [59/100] - Train Loss: 1.1085 Val Loss: 1.1789 - Train Acc: 65.25% - Val Acc: 57.05%
# Epoch [60/100] - Train Loss: 1.1071 Val Loss: 1.1782 - Train Acc: 64.90% - Val Acc: 57.05%
# Epoch [61/100] - Train Loss: 1.1116 Val Loss: 1.1779 - Train Acc: 65.04% - Val Acc: 57.05%
# Epoch [62/100] - Train Loss: 1.1091 Val Loss: 1.1781 - Train Acc: 64.75% - Val Acc: 57.05%
# Epoch [63/100] - Train Loss: 1.1092 Val Loss: 1.1778 - Train Acc: 64.75% - Val Acc: 57.05%
# Epoch [64/100] - Train Loss: 1.1093 Val Loss: 1.1774 - Train Acc: 64.25% - Val Acc: 57.05%
# Epoch [65/100] - Train Loss: 1.1063 Val Loss: 1.1778 - Train Acc: 64.25% - Val Acc: 57.38%
# Epoch [66/100] - Train Loss: 1.0984 Val Loss: 1.1782 - Train Acc: 64.03% - Val Acc: 57.05%
# Epoch [67/100] - Train Loss: 1.1058 Val Loss: 1.1778 - Train Acc: 63.82% - Val Acc: 57.86%
# Epoch [68/100] - Train Loss: 1.1092 Val Loss: 1.1778 - Train Acc: 64.39% - Val Acc: 57.86%
# Epoch [69/100] - Train Loss: 1.1096 Val Loss: 1.1768 - Train Acc: 64.32% - Val Acc: 57.86%
# Epoch [70/100] - Train Loss: 1.1005 Val Loss: 1.1766 - Train Acc: 64.68% - Val Acc: 57.53%
# Epoch [71/100] - Train Loss: 1.1063 Val Loss: 1.1764 - Train Acc: 64.61% - Val Acc: 57.05%
# Epoch [72/100] - Train Loss: 1.1043 Val Loss: 1.1760 - Train Acc: 65.04% - Val Acc: 57.38%
# Epoch [73/100] - Train Loss: 1.1061 Val Loss: 1.1753 - Train Acc: 63.68% - Val Acc: 57.38%
# Epoch [74/100] - Train Loss: 1.1027 Val Loss: 1.1747 - Train Acc: 65.04% - Val Acc: 57.38%
# Epoch [75/100] - Train Loss: 1.1009 Val Loss: 1.1747 - Train Acc: 64.68% - Val Acc: 57.05%
# Epoch [76/100] - Train Loss: 1.0985 Val Loss: 1.1749 - Train Acc: 64.97% - Val Acc: 57.05%
# Epoch [77/100] - Train Loss: 1.1015 Val Loss: 1.1751 - Train Acc: 63.68% - Val Acc: 57.38%
# Epoch [78/100] - Train Loss: 1.0996 Val Loss: 1.1743 - Train Acc: 64.82% - Val Acc: 57.38%
# Epoch [79/100] - Train Loss: 1.0987 Val Loss: 1.1742 - Train Acc: 65.61% - Val Acc: 57.38%
# Epoch [80/100] - Train Loss: 1.1043 Val Loss: 1.1736 - Train Acc: 63.96% - Val Acc: 57.86%
# Epoch [81/100] - Train Loss: 1.0969 Val Loss: 1.1733 - Train Acc: 65.18% - Val Acc: 57.86%
# Epoch [82/100] - Train Loss: 1.0982 Val Loss: 1.1733 - Train Acc: 64.82% - Val Acc: 58.53%
# Epoch [83/100] - Train Loss: 1.0971 Val Loss: 1.1741 - Train Acc: 65.11% - Val Acc: 58.53%
# Epoch [84/100] - Train Loss: 1.0958 Val Loss: 1.1738 - Train Acc: 64.90% - Val Acc: 58.86%
# Epoch [85/100] - Train Loss: 1.1020 Val Loss: 1.1737 - Train Acc: 64.18% - Val Acc: 57.86%
# Epoch [86/100] - Train Loss: 1.0969 Val Loss: 1.1745 - Train Acc: 64.47% - Val Acc: 58.86%
# Epoch [87/100] - Train Loss: 1.0986 Val Loss: 1.1733 - Train Acc: 64.54% - Val Acc: 58.53%
# Epoch [88/100] - Train Loss: 1.0960 Val Loss: 1.1732 - Train Acc: 65.40% - Val Acc: 58.86%
# Epoch [89/100] - Train Loss: 1.1041 Val Loss: 1.1734 - Train Acc: 63.89% - Val Acc: 57.05%
# Epoch [90/100] - Train Loss: 1.0933 Val Loss: 1.1729 - Train Acc: 65.61% - Val Acc: 57.38%
# Epoch [91/100] - Train Loss: 1.0939 Val Loss: 1.1729 - Train Acc: 64.90% - Val Acc: 57.72%
# Epoch [92/100] - Train Loss: 1.0984 Val Loss: 1.1724 - Train Acc: 65.18% - Val Acc: 57.72%
# Epoch [93/100] - Train Loss: 1.1000 Val Loss: 1.1729 - Train Acc: 64.39% - Val Acc: 57.05%
# Epoch [94/100] - Train Loss: 1.0943 Val Loss: 1.1729 - Train Acc: 65.04% - Val Acc: 58.53%
# Epoch [95/100] - Train Loss: 1.0941 Val Loss: 1.1734 - Train Acc: 65.47% - Val Acc: 57.86%
# Epoch [96/100] - Train Loss: 1.0962 Val Loss: 1.1731 - Train Acc: 64.03% - Val Acc: 57.05%
# Epoch [97/100] - Train Loss: 1.0849 Val Loss: 1.1731 - Train Acc: 66.04% - Val Acc: 58.86%
# Epoch [98/100] - Train Loss: 1.0875 Val Loss: 1.1733 - Train Acc: 65.69% - Val Acc: 58.53%
# Epoch [99/100] - Train Loss: 1.0901 Val Loss: 1.1726 - Train Acc: 65.54% - Val Acc: 57.86%
# Epoch [100/100] - Train Loss: 1.0962 Val Loss: 1.1724 - Train Acc: 65.18% - Val Acc: 57.38%

        training_history = history(loss = [1.3831, 1.3795, 1.3750, 1.3680, 1.3628, 1.3574, 1.3510, 1.3431, 1.3321, 1.3245, 1.3136, 1.3058, 1.2989, 1.2888, 1.2815, 1.2721, 1.2637, 1.2548, 1.2511, 
                                           1.2404, 1.2319, 1.2265, 1.2167, 1.2084, 1.2000, 1.2004, 1.1915, 1.1843, 1.1821, 1.1758, 1.1696, 1.1664, 1.1576, 1.1561, 1.1559, 1.1477, 1.1480, 1.1437, 
                                           1.1427, 1.1379, 1.1348, 1.1349, 1.1317, 1.1302, 1.1277, 1.1309, 1.1261, 1.1264, 1.1238, 1.1149, 1.1195, 1.1149, 1.1204, 1.1104, 1.1172, 1.1169, 1.1143,
                                           1.1121, 1.1085, 1.1071, 1.1116, 1.1091, 1.1092, 1.1093, 1.1063, 1.0984, 1.1058, 1.1092, 1.1096, 1.1005, 1.1063, 1.1043, 1.1061, 1.1027, 1.1009, 1.0985,
                                           1.1015, 1.0996, 1.0987, 1.1043, 1.0969, 1.0982, 1.0971, 1.0958, 1.1020, 1.0969, 1.0986, 1.0960, 1.1091, 1.0933, 1.0939, 1.0984, 1.1000, 1.0943, 1.0941,
                                           1.0962, 1.0849, 1.0875, 1.0901, 1.0962],
                                   val_loss = [1.3825, 1.3785, 1.3741, 1.3691, 1.3635, 1.3572, 1.3504, 1.3432, 1.3352, 1.3271, 1.3186, 1.3101, 1.3022, 1.2944, 1.2870, 1.2800, 1.2731, 1.2667, 1.2606,
                                               1.2545, 1.2491, 1.2438, 1.2383, 1.2332, 1.2285, 1.2243, 1.2204, 1.2167, 1.2136, 1.2108, 1.2082, 1.2056, 1.2035, 1.2017, 1.2000, 1.1981, 1.1965, 1.1950,
                                               1.1934, 1.1923, 1.1907, 1.1897, 1.1888, 1.1880, 1.1871, 1.1859, 1.1852, 1.1846, 1.1842, 1.1841, 1.1833, 1.1821, 1.1817, 1.1810, 1.1803, 1.1799, 1.1797,
                                               1.1790, 1.1789, 1.1781, 1.1779, 1.1778, 1.1774, 1.1778, 1.1778, 1.1768, 1.1766, 1.1764, 1.1760, 1.1753, 1.1747, 1.1747, 1.1749, 1.1741, 1.1736, 1.1733,
                                               1.1733, 1.1734, 1.1738, 1.1745, 1.1731, 1.1731, 1.1733, 1.1726, 1.1729, 1.1729, 1.1724, 1.1729, 1.1734, 1.1731, 1.1729, 1.1724, 1.1731, 1.1733, 1.1726,
                                               1.1729, 1.1733, 1.1731, 1.1729, 1.1724],
                                   accuracy = [26.70, 32.38, 34.67, 35.82, 37.33, 38.12, 38.12, 40.63, 41.42, 42.93, 43.29, 44.58, 44.87, 45.66, 44.36, 44.94, 46.88, 47.95, 49.39, 50.75,
                                               53.05, 54.27, 56.07, 57.14, 57.50, 58.72, 59.58, 60.52, 60.37, 60.09, 62.17, 60.88, 62.60, 63.32, 62.89, 63.68, 62.67, 63.96, 64.03, 63.82, 64.54,
                                               64.39, 64.47, 64.75, 65.04, 64.25, 65.18, 64.75, 64.32, 64.25, 65.04, 65.25, 64.61, 64.54, 64.90, 65.11, 65.18, 64.39, 64.82, 65.61, 65.69, 65.54,
                                               65.25, 64.68, 64.97, 65.18, 64.68, 65.47, 64.03, 66.04, 65.69, 65.54, 64.18, 65.61, 65.04, 64.47, 64.82, 63.89, 65.61, 65.69, 64.39, 65.04, 65.47, 
                                               64.25, 64.82, 65.18, 63.96, 65.18, 64.03, 66.04, 65.69, 65.54, 64.18, 65.61, 65.04, 64.47, 64.82 ,63.89, 65.61, 65.69],
                                   val_accuracy = [33.22, 45.64, 45.64, 45.64, 46.31, 46.31, 45.97, 45.64, 45.64, 45.30, 45.30, 45.64, 45.30, 45.97, 47.32, 46.98, 48.99, 49.33, 50.34, 51.01,
                                                   51.34, 51.34, 50.67, 51.68, 52.35, 53.69, 53.69, 54.36, 54.36, 55.03, 54.70, 55.03, 55.37, 55.70, 55.37, 56.38, 56.38, 56.71, 57.05, 56.71,
                                                   56.04, 56.38, 55.37, 56.04, 56.04, 56.04, 56.71, 57.05, 57.05, 57.05, 56.71, 56.71, 56.71, 56.71, 57.05, 57.05, 57.05, 57.05, 57.05, 57.38, 
                                                   57.05, 57.86, 57.86, 57.86, 57.38, 57.38, 57.38, 57.05, 57.05, 57.38, 57.38, 57.38, 57.86, 57.86, 57.86, 57.05, 57.38, 57.38, 57.38, 57.86, 
                                                   57.86, 57.86, 57.05, 58.86, 58.53, 57.86, 57.38, 58.86, 58.53, 58.86, 58.86, 57.05, 58.53, 58.53, 58.86, 58.86, 57.05, 57.38, 57.72, 57.72])
                                                

        print(len(training_history.history['loss']))
        print(len(training_history.history['val_loss']))
        print(len(training_history.history['accuracy']))
        print(len(training_history.history['val_accuracy']))

        output_path = os.path.join('models', 'ensemble')
        plot_acc_loss(training_history, output_path)


    else:
        print("Use --cm to draw confusion matrix")