import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path += ["../utils/", "./audio/implementation", "./audio"]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from draw_plot import plot_acc_loss_torch
from CustomSpectrogramSortedDataset import CustomSpectrogramSortedDataset
from train_sarkar_torch import SarkarVGGCustomizedArchitecture
from EnsembleModel import EnsembleModel
from torchsummary import summary


def save_checkpoint(model, path, current_accuracy, epoch):
    if current_accuracy > 56.: 
        torch.save(model.state_dict(), os.path.join(path, f"joint_{current_accuracy}_{epoch}.pth"))


def train_network(path, batch_size, l2_lambda, learning_rate, epochs, img_height, img_width):
    NUM_CLASSES = 4
    CHANNELS = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #define models
    model_audio = SarkarVGGCustomizedArchitecture(NUM_CLASSES, CHANNELS).to(device)
    model_audio.load_state_dict(torch.load("./audio/trained_models/torch/checkpoints7/sarkar_57.53_445.pth"))

    #to be corrected
    model_lyrics = SarkarVGGCustomizedArchitecture(NUM_CLASSES, CHANNELS).to(device)
    model_lyrics.load_state_dict(torch.load("./audio/trained_models/torch/checkpoints7/sarkar_57.53_445.pth"))
    
    #load ensemble model
    ensembleModel = EnsembleModel(model_audio, model_lyrics, NUM_CLASSES).to(device)
    optimizer = optim.AdamW(ensembleModel.parameters(), lr=learning_rate, amsgrad=False)#, weight_decay=l2_lambda)
    criterion = nn.CrossEntropyLoss()
    
    #train
    val_accuracy_history = []
    val_loss_history = []
        
    transform = ToTensor()
    #load audio models
    train_audio_dataset = CustomSpectrogramSortedDataset(os.path.join(path, "train"), transform=transform)
    val_audio_dataset = CustomSpectrogramSortedDataset(os.path.join(path, "val"), transform=transform)
    train_audio_loader = DataLoader(train_audio_dataset, batch_size=batch_size, shuffle=True)
    val_audio_loader = DataLoader(val_audio_dataset, batch_size=batch_size, shuffle=False)
    
    #load lyrics models
    train_lyrics_dataset = CustomSpectrogramSortedDataset(os.path.join(path, "train"), transform=transform)
    val_lryics_dataset = CustomSpectrogramSortedDataset(os.path.join(path, "val"), transform=transform)
    train_lyrics_loader = DataLoader(train_audio_dataset, batch_size=batch_size, shuffle=True)
    val_lyrics_loader = DataLoader(val_audio_dataset, batch_size=batch_size, shuffle=False)
    
    # summary(ensembleModel) #, input_size=(CHANNELS, img_height, img_width))
    
    for epoch in range(epochs):
        ensembleModel.train()
        train_loss = 0.0
        
        for (inputs_audio, labels_audio), (inputs_lyrics, labels_lyrics) in zip(train_audio_loader, train_lyrics_loader):
            inputs_audio = inputs_audio.to(device)
            inputs_lyrics = inputs_lyrics.to(device)
            labels_audio = torch.argmax(labels_audio, dim=1).to(device)
            labels_lyrics = torch.argmax(labels_lyrics, dim=1).to(device)  # Convert to class indices from one hot encoding if needed

            optimizer.zero_grad()
            outputs = ensembleModel(inputs_audio, inputs_lyrics)
            loss = criterion(outputs, labels_audio) #for example audio, it really shouldnt matter
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs_audio.size(0)
        
        ensembleModel.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
    #     with torch.no_grad():
    #         for inputs, labels in val_loader:
    #             inputs = inputs.to(device)
    #             labels = torch.argmax(labels, dim=1).to(device)
                
    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)
    #             val_loss += loss.item() * inputs.size(0)
    #             _, predicted = outputs.max(1)
    #             total += labels.size(0)
    #             correct += predicted.eq(labels).sum().item()
        
    #     val_accuracy = 100 * correct / total
    #     val_loss_history.append(val_loss)
    #     val_accuracy_history.append(val_accuracy)
        
    #     print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss / len(train_loader.dataset):.4f} - "
    #           f"Val Loss: {val_loss / len(val_loader.dataset):.4f} - Val Acc: {val_accuracy:.2f}%")

    #     checkpoint_path = "./trained_models/torch/checkpoints3/"
    #     os.makedirs(checkpoint_path, exist_ok=True)
    #     save_checkpoint(model, checkpoint_path, val_accuracy, epoch+1)
        
    # model_path = f"./trained_models/torch/sarkar_approach3_{path[-42:]}_{epochs}_{val_accuracy:.2f}.pth"
    # torch.save(model.state_dict(), model_path)
    # plot_acc_loss_torch(val_accuracy_history, val_loss_history, "./histories/torch/history_500_AdamW_approach3")
    
    
if __name__ == "__main__":
    path = "../database/melgrams/gray/different-params/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray" 
    NUM_EPOCHS = 500
    BATCH_SIZE = 16
    L2_LAMBDA = 1e-3
    LEARNING_RATE = 1e-5
    IM_WIDTH = 1292
    IM_HEIGHT = 128
    
    train_network(path=path, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, l2_lambda=L2_LAMBDA, 
                  epochs=NUM_EPOCHS, img_width=IM_WIDTH, img_height=IM_HEIGHT)
    