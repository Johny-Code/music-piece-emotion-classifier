import matplotlib.pyplot as plt
import seaborn as sns

def draw_confusion_matrix(cm, target_names, cmap=None, output_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()