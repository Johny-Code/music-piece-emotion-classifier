import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
sys.path += ["../../utils/", "./implementation"]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from contextlib import redirect_stdout
from sklearn.metrics import classification_report, confusion_matrix
from draw_plot import plot_acc_loss_torch, draw_confusion_matrix
from CustomSpectrogramDataset import CustomSpectrogramDataset
from train_sarkar_torch import SarkarVGGCustomizedArchitecture


def validate_model(model_path, dataset_path, img_height, img_width, label_names, metrics_file, confusion_matrix_prefix):
    NUM_CLASSES = 4
    CHANNELS = 1
    BATCH_SIZE = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SarkarVGGCustomizedArchitecture(NUM_CLASSES, CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = ToTensor()
    test_dataset = CustomSpectrogramDataset(os.path.join(dataset_path, "test"), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    summary(model, input_size=(CHANNELS, img_height, img_width))
    
    correct = 0
    total = 0
    all_predicted_labels = []
    all_actual_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = torch.argmax(labels, dim=1).to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_actual_labels.append(predicted.cpu().numpy())
            all_predicted_labels.append(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
                
    test_accuracy = 100 * correct / total
    
    flattened_predicted = np.concatenate(all_predicted_labels).ravel()
    flattened_true = np.concatenate(all_actual_labels).ravel()

    cfm = confusion_matrix(flattened_true, flattened_predicted)
    draw_confusion_matrix(cfm, label_names, "confusion_matrices/torch/", 
                          filename_prefix=f"{confusion_matrix_prefix}_{model_path[-13:-8]}_{test_accuracy:.2f}")

    os.makedirs("./metrics", exist_ok=True)
    with open(os.path.join("metrics/torch", metrics_file + f"_{model_path[-13:-8]}_{test_accuracy:.2f}"),'w') as file:
        with redirect_stdout(file):
            print(classification_report(flattened_true, flattened_predicted, target_names=label_names, digits=4))
    
    print(classification_report(flattened_true, flattened_predicted, target_names=label_names, digits=4))
    
    
if __name__ == "__main__":
    dataset_path = "../../database/melgrams/gray/different-params/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray"
    model_path = "./trained_models/torch/checkpoints7/sarkar_56.86_473.pth"
    name = "torch_checkpoint7_copy9"
    label_names = ["happy", "angry", "sad", "relaxed"]
    IM_WIDTH = 1292
    IM_HEIGHT = 128
    
    validate_model(model_path=model_path, dataset_path=dataset_path, img_width=IM_WIDTH, 
                   img_height=IM_HEIGHT, label_names=label_names, metrics_file=f"{name}.txt", 
                   confusion_matrix_prefix=name)
