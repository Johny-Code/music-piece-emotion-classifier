import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path += ["../../utils/", "./implementation"]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from draw_plot import plot_acc_loss_torch
from CustomSpectrogramDataset import CustomSpectrogramDataset
from torchsummary import summary


class SarkarVGGCustomizedArchitecture(nn.Module):
    def __init__(self, num_classes, channels):
        super(SarkarVGGCustomizedArchitecture, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout(0.25),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2816, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def save_checkpoint(model, path, highest_accuracy, current_accuracy, epoch):
    if current_accuracy > 50. and (current_accuracy > highest_accuracy): 
        torch.save(model.state_dict(), os.path.join(path, f"sarkar_{current_accuracy}_{epoch}.pth"))


def train_network(path, batch_size, l2_lambda, learning_rate, epochs, img_height, img_width):
    NUM_CLASSES = 4
    CHANNELS = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SarkarVGGCustomizedArchitecture(NUM_CLASSES, CHANNELS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)#, weight_decay=l2_lambda)
    criterion = nn.CrossEntropyLoss()
    val_accuracy_history = []
    val_loss_history = []
        
    transform = ToTensor()
    train_dataset = CustomSpectrogramDataset(os.path.join(path, "train"), transform=transform)
    val_dataset = CustomSpectrogramDataset(os.path.join(path, "val"), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    summary(model, input_size=(CHANNELS, img_height, img_width))
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        highest_accuracy = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = torch.argmax(labels, dim=1).to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_accuracy = 100 * correct / total
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss / len(train_loader.dataset):.4f} - "
              f"Val Loss: {val_loss / len(val_loader.dataset):.4f} - Val Acc: {val_accuracy:.2f}%")

        checkpoint_path = "./trained_models/torch/checkpoints4/"
        os.makedirs(checkpoint_path, exist_ok=True)
        save_checkpoint(model, checkpoint_path, highest_accuracy+0.01, val_accuracy, epoch+1)
        highest_accuracy = val_accuracy if val_accuracy > highest_accuracy else highest_accuracy
        
    model_path = f"./trained_models/torch/sarkar_AdamW_{path[-42:]}_{epochs}_{val_accuracy:.2f}.pth"
    torch.save(model.state_dict(), model_path)
    plot_acc_loss_torch(val_accuracy_history, val_loss_history, "./histories/torch/history_500_AdamW_batch16")
    
    
if __name__ == "__main__":
    path = "../../database/melgrams/gray/different-params/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray" 
    NUM_EPOCHS = 500
    BATCH_SIZE = 16
    L2_LAMBDA = 1e-3
    LEARNING_RATE = 1e-5
    IM_WIDTH = 1292
    IM_HEIGHT = 128
    
    train_network(path=path, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, l2_lambda=L2_LAMBDA, 
                  epochs=NUM_EPOCHS, img_width=IM_WIDTH, img_height=IM_HEIGHT)
    #try weight_decay=1e-3 and amsgrad=True