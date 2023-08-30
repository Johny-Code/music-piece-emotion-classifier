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
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout1 = nn.Dropout(0.25)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout2 = nn.Dropout(0.25)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout3 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2816, 256)
        self.relu8 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.relu9 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        x = self.dropout1(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool4(x)
        x = self.dropout2(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.maxpool5(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu8(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.relu9(x)
        x = self.dropout5(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x
    

def save_checkpoint(model, path, current_accuracy, epoch):
    if current_accuracy > 56.: 
        torch.save(model.state_dict(), os.path.join(path, f"sarkar_{current_accuracy}_{epoch}.pth"))


def train_network(path, batch_size, l2_lambda, learning_rate, epochs, img_height, img_width):
    NUM_CLASSES = 4
    CHANNELS = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SarkarVGGCustomizedArchitecture(NUM_CLASSES, CHANNELS).to(device)
    optimizer = optim.AdamW(model.parameters() ,lr=learning_rate, amsgrad=False)#, weight_decay=l2_lambda)
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

        checkpoint_path = "./trained_models/torch/checkpoints7/"
        os.makedirs(checkpoint_path, exist_ok=True)
        save_checkpoint(model, checkpoint_path, val_accuracy, epoch+1)
        
    model_path = f"./trained_models/torch/sarkar_approach7_{path[-42:]}_{epochs}_{val_accuracy:.2f}.pth"
    torch.save(model.state_dict(), model_path)
    plot_acc_loss_torch(val_accuracy_history, val_loss_history, "./histories/torch/history_500_AdamW_approach7")
    
    
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
    