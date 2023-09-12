import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import sys
sys.path += ["../utils/", "../../utils", "./audio/implementation", "./audio/implementation/dataset", "./audio", "./lyric",
             "./lyric/implementation", "./lyric/implementation/dataset", "./../tools/"]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.transforms import ToTensor
from draw_plot import plot_acc_loss_torch, draw_confusion_matrix
from CustomSpectrogramSortedDataset import CustomSpectrogramSortedDataset
from CustomLyricSortedTensorDataset import CustomLyricSortedTensorDataset
from CustomXLNetForMultiLabelSequenceClassification import CustomXLNetForMultiLabelSequenceClassification
from train_sarkar_torch import SarkarVGGCustomizedArchitecture
from EnsembleModel import EnsembleModel
from torchsummary import summary
from sklearn.metrics import classification_report, confusion_matrix
from train_xlnet import load_model, load_dataset, tokenize_inputs, create_attention_masks, to_tensor
from transformers import XLNetTokenizer, XLNetModel, AdamW
from contextlib import redirect_stdout


def validate_model(model_path, audio_model_path, lyric_model_path, lyric_dataset_path, database_path,
                   confusion_matrix_prefix, label_names, metrics_file, audio_dataset_path, hyperparameters):
    NUM_CLASSES = 4
    CHANNELS = 1
    BATCH_SIZE = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #define models
    model_audio = SarkarVGGCustomizedArchitecture(NUM_CLASSES, CHANNELS).to(device)
    model_audio.load_state_dict(torch.load(audio_model_path))    
    custom_xlnet_model = CustomXLNetForMultiLabelSequenceClassification()
    pretrained_model_state_dict = torch.load(lyric_model_path)
    
    #load ensemble model
    ensembleModel = EnsembleModel(model_audio, custom_xlnet_model, NUM_CLASSES, BATCH_SIZE).to(device)
    model = ensembleModel
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = ToTensor()
    criterion = nn.CrossEntropyLoss()
    
    #load audio models
    audio_test_dataset = CustomSpectrogramSortedDataset(os.path.join(audio_dataset_path, "test"), transform=transform)
    audio_test_loader = DataLoader(audio_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    #load lyrics models 
    _, test_dataset, _ = load_dataset(lyric_dataset_path, database_path, load_full_dataset=True)
    tokienizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case = hyperparameters['tokenizer']['do_lower_case'])
    
    test_labels = np.array(test_dataset['mood'].tolist())    
    test_input_ids = tokenize_inputs(hyperparameters, test_dataset['lyric'].tolist(), tokienizer)
    test_attention_masks  = create_attention_masks(test_input_ids)

    test_input_ids, test_attention_masks, test_labels = to_tensor(test_input_ids, test_attention_masks, test_labels)
    
    lyric_test_dataset = CustomLyricSortedTensorDataset(test_input_ids, test_attention_masks, test_labels,
                                                       labels_names=test_dataset['mood'].index.values.tolist())
    
    lyric_test_loader = DataLoader(lyric_test_dataset, sampler=SequentialSampler(lyric_test_dataset), #Random Sampler breaks correct order
                            batch_size=hyperparameters['model']['batch_size'], shuffle=False)
        
    summary(model)
    
    correct = 0
    total = 0
    all_predicted_labels = []
    all_actual_labels = []
    
    assert len(lyric_test_dataset) == len(audio_test_dataset), "Lyric and audio datasets have different lenghts"
    
    with torch.no_grad():         
        for (inputs_audio, labels_audio_tensor, labels_audio), (batch_lyrics, labels_lyrics) in zip(audio_test_loader, lyric_test_loader):
            inputs_audio = inputs_audio.to(device)
            labels_audio_tensor = torch.argmax(labels_audio_tensor, dim=1).to(device)
            
            assert labels_audio[0] == labels_lyrics[0], "Different labels are compared"
                        
            batch = tuple(t.to(device) for t in batch_lyrics)
            b_input_ids, b_input_mask, b_labels = batch

            outputs = ensembleModel(inputs_audio, input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            _, predicted = outputs.max(1)
            
            all_predicted_labels.append(predicted.cpu().numpy())
            all_actual_labels.append(labels_audio_tensor.cpu().numpy())
            
            total += labels_audio_tensor.size(0)
            correct += predicted.eq(labels_audio_tensor).sum().item()

    test_accuracy = 100 * correct / total
    
    flattened_predicted = np.concatenate(all_predicted_labels).ravel()
    flattened_true = np.concatenate(all_actual_labels).ravel()

    cfm = confusion_matrix(flattened_true, flattened_predicted)
    draw_confusion_matrix(cfm, label_names, "confusion_matrices/", 
                          filename_prefix=f"{confusion_matrix_prefix}_{test_accuracy:.2f}")

    os.makedirs("./metrics", exist_ok=True)
    with open(os.path.join("metrics", metrics_file + f"_{test_accuracy:.2f}"),'w') as file:
        with redirect_stdout(file):
            print(classification_report(flattened_true, flattened_predicted, target_names=label_names, digits=4))
    
    print(classification_report(flattened_true, flattened_predicted, target_names=label_names, digits=4))
    
    
if __name__ == "__main__":
    model_path = "./trained_models/copy2_joint_58.33_lyrics16_256x2_audio_256_last_5_no_dropout_7.pth"
    name = "joint_concatenated_copy10"
    database_path = "../database/MoodyLyrics4Q_cleaned_split.csv"
    lyrics_dataset_path = "../database/lyrics"
    audio_dataset_path = "../database/melgrams/gray/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray_middle30s_corrected" 
    audio_model_path = "./audio/trained_models/torch/checkpoints7/sarkar_57.53_445.pth"
    lyric_model_path = "./lyric/xlnet/xlnet_2023-09-01_23-29-57.pt"    
    label_names = ["happy", "angry", "sad", "relaxed"]
    hyperparameters = {
                            'tokenizer':{
                                'do_lower_case': True,
                                'num_embeddings': 256,
                            },
                            'model':{
                                'num_labels': 4,
                                'batch_size': 1,
                            }
                        }
    
    validate_model(model_path=model_path, audio_model_path=audio_model_path, lyric_model_path=lyric_model_path,
                   lyric_dataset_path=lyrics_dataset_path, database_path=database_path, confusion_matrix_prefix=name, 
                   label_names=label_names, metrics_file=name, audio_dataset_path=audio_dataset_path, 
                   hyperparameters=hyperparameters)
