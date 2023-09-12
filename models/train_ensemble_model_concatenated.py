import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import sys
sys.path += ["../utils/", "./audio/implementation", "./audio/implementation/dataset", "./audio", "./lyric",
             "./lyric/implementation", "./lyric/implementation/dataset", "./../tools/"]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import ToTensor
from draw_plot import plot_acc_loss_torch
from CustomSpectrogramSortedDataset import CustomSpectrogramSortedDataset
from CustomLyricSortedTensorDataset import CustomLyricSortedTensorDataset
from CustomXLNetForMultiLabelSequenceClassification import CustomXLNetForMultiLabelSequenceClassification
from train_sarkar_torch import SarkarVGGCustomizedArchitecture
from EnsembleModel import EnsembleModel
from torchsummary import summary
from train_xlnet import load_model, load_dataset, tokenize_inputs, create_attention_masks, to_tensor
from transformers import XLNetTokenizer, XLNetModel, AdamW


def save_checkpoint(model, path, current_accuracy, epoch):
    if current_accuracy > 58.: 
        torch.save(model.state_dict(), os.path.join(path, f"joint_{current_accuracy}_{epoch}.pth"))


def train_network(path, batch_size, l2_lambda, learning_rate, epochs, img_height, img_width, hyperparameters, nb_classes, channels,
                  audio_model_path, lyric_model_path, database_path, lyrics_dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_accuracy_history = []
    val_loss_history = []
    
    #define models
    model_audio = SarkarVGGCustomizedArchitecture(nb_classes, channels).to(device)
    model_audio.load_state_dict(torch.load(audio_model_path))    
    custom_xlnet_model = CustomXLNetForMultiLabelSequenceClassification()
    pretrained_model_state_dict = torch.load(lyric_model_path)

    # Copy the weights from the pre-trained model's state_dict() to the custom model
    custom_model_state_dict = custom_xlnet_model.state_dict()
    for name, param in pretrained_model_state_dict.items():
        if name in custom_model_state_dict and param.shape == custom_model_state_dict[name].shape:
            custom_model_state_dict[name].copy_(param)
    custom_xlnet_model.load_state_dict(custom_model_state_dict)
        
    #load ensemble model
    ensembleModel = EnsembleModel(model_audio, custom_xlnet_model, nb_classes, BATCH_SIZE).to(device)
    optimizer = optim.AdamW(ensembleModel.parameters(), lr=learning_rate, amsgrad=False)#, weight_decay=l2_lambda)
    
    criterion = nn.CrossEntropyLoss()
    transform = ToTensor()
    
    #load audio models
    audio_train_dataset = CustomSpectrogramSortedDataset(os.path.join(path, "train"), transform=transform)
    audio_val_dataset = CustomSpectrogramSortedDataset(os.path.join(path, "val"), transform=transform)
    audio_train_loader = DataLoader(audio_train_dataset, batch_size=batch_size, shuffle=False)
    audio_val_loader = DataLoader(audio_val_dataset, batch_size=batch_size, shuffle=False)
    
    #load lyrics models 
    train_dataset, _, val_dataset = load_dataset(lyrics_dataset_path, database_path, load_full_dataset=True)
    tokienizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case = hyperparameters['tokenizer']['do_lower_case'])
    
    train_labels = np.array(train_dataset['mood'].tolist())    
    train_input_ids = tokenize_inputs(hyperparameters, train_dataset['lyric'].tolist(), tokienizer)
    train_attention_masks  = create_attention_masks(train_input_ids)

    val_labels = np.array(val_dataset['mood'].tolist())
    val_input_ids = tokenize_inputs(hyperparameters, val_dataset['lyric'].tolist(), tokienizer)
    val_attention_masks  = create_attention_masks(val_input_ids)

    train_input_ids, train_attention_masks, train_labels = to_tensor(train_input_ids, train_attention_masks, train_labels)
    val_input_ids, val_attention_masks, val_labels = to_tensor(val_input_ids, val_attention_masks, val_labels)
    
    lyric_train_dataset = CustomLyricSortedTensorDataset(train_input_ids, train_attention_masks, train_labels,
                                                         labels_names=train_dataset['mood'].index.values.tolist())
    lyric_val_dataset = CustomLyricSortedTensorDataset(val_input_ids, val_attention_masks, val_labels,
                                                       labels_names=val_dataset['mood'].index.values.tolist())
    
    lyric_train_loader = DataLoader(lyric_train_dataset, sampler=SequentialSampler(lyric_train_dataset), 
                            batch_size=hyperparameters['model']['batch_size'], shuffle=False)
    lyric_val_loader = DataLoader(lyric_val_dataset, sampler=SequentialSampler(lyric_val_dataset), 
                            batch_size=hyperparameters['model']['batch_size'], shuffle=False)
    
    summary(ensembleModel)
    
    for epoch in range(epochs):
        ensembleModel.train()
        train_loss = 0.0
        
        for (inputs_audio, labels_audio_tensor, labels_audio), (batch_lyrics, labels_lyrics) in zip(audio_train_loader, lyric_train_loader):
            inputs_audio = inputs_audio.to(device)
            labels_audio_tensor = torch.argmax(labels_audio_tensor, dim=1).to(device)
            
            assert labels_audio[0] == labels_lyrics[0], "Different labels are compared"
            
            batch = tuple(t.to(device) for t in batch_lyrics)
            b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad()
            outputs = ensembleModel(inputs_audio, input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                        
            loss = criterion(outputs, labels_audio_tensor) #for example audio, it really shouldnt matter
            loss.backward()
            optimizer.step()
                        
            train_loss += loss.item() * inputs_audio.size(0)
            
                     
        ensembleModel.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0  
        
        with torch.no_grad():         
            for (inputs_audio, labels_audio_tensor, labels_audio), (batch_lyrics, labels_lyrics) in zip(audio_val_loader, lyric_val_loader):
                inputs_audio = inputs_audio.to(device)
                labels_audio_tensor = torch.argmax(labels_audio_tensor, dim=1).to(device)
                
                assert labels_audio[0] == labels_lyrics[0], "Different labels are compared"
                
                batch = tuple(t.to(device) for t in batch_lyrics)
                b_input_ids, b_input_mask, b_labels = batch

                outputs = ensembleModel(inputs_audio, input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = criterion(outputs, labels_audio_tensor)
                val_loss += loss.item() * inputs_audio.size(0)
                _, predicted = outputs.max(1)
                total += labels_audio_tensor.size(0)
                correct += predicted.eq(labels_audio_tensor).sum().item()            
        
        val_accuracy = 100 * correct / total
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss / len(audio_train_loader.dataset):.4f} - "
              f"Val Loss: {val_loss / len(audio_train_loader.dataset):.4f} - Val Acc: {val_accuracy:.2f}%")

        checkpoint_path = "./trained_models/new/"
        os.makedirs(checkpoint_path, exist_ok=True)
        save_checkpoint(ensembleModel, checkpoint_path, val_accuracy, epoch+1)
    
    
if __name__ == "__main__":
    database_path = "../database/MoodyLyrics4Q_cleaned_split.csv"
    lyrics_dataset_path = "../database/lyrics"
    audio_dataset_path = "../database/melgrams/gray/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray_middle30s_corrected" 
    audio_model_path = "./audio/trained_models/torch/checkpoints7/sarkar_57.53_445.pth"
    lyric_model_path = "./lyric/xlnet/xlnet_2023-09-01_23-29-57.pt"
    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    L2_LAMBDA = 1e-3
    LEARNING_RATE = 1e-5
    NUM_EMBEDDINGS = 256
    IM_WIDTH = 1292
    IM_HEIGHT = 128
    CHANNELS = 1
    NB_CLASSES = 4
    HYPERPARAMETERS = {
                            'tokenizer':{
                                'do_lower_case': True,
                                'num_embeddings': NUM_EMBEDDINGS,
                            },
                            'model':{
                                'num_labels': NB_CLASSES,
                                'batch_size': BATCH_SIZE,
                            }
                        }
    
    train_network(path=audio_dataset_path, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, l2_lambda=L2_LAMBDA, 
                  epochs=NUM_EPOCHS, img_width=IM_WIDTH, img_height=IM_HEIGHT, hyperparameters=HYPERPARAMETERS,
                  nb_classes=NB_CLASSES, channels=CHANNELS, audio_model_path=audio_model_path, lyric_model_path=lyric_model_path,
                  database_path=database_path, lyrics_dataset_path=lyrics_dataset_path)
    