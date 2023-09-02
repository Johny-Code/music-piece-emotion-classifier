import os
import sys
import operator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
sys.path += ["../utils/", "./audio/implementation", "./audio/implementation/dataset", "./lyric/implementation/" ,
             "./lyric/implementation/dataset", "./audio", "./lyric", "./../tools/"]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from contextlib import redirect_stdout
from sklearn.metrics import classification_report, confusion_matrix
from draw_plot import plot_acc_loss_torch, draw_confusion_matrix
from CustomSpectrogramDatasetWithPaths import CustomSpectrogramDatasetWithPaths
from CustomLyricTensorDataset import CustomLyricTensorDataset
from train_sarkar_torch import SarkarVGGCustomizedArchitecture
from transformers import XLNetTokenizer, XLNetModel, AdamW
from train_xlnet import load_dataset, to_tensor, to_custom_tensorDataset_dataLoader_tuple, load_model, tokenize_inputs
from train_xlnet import create_attention_masks


def validate_audio_model(model_path, dataset_path, img_height, img_width, label_names, metrics_file, 
                         confusion_matrix_prefix):
    NUM_CLASSES = 4
    CHANNELS = 1
    BATCH_SIZE = 1
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_audio = SarkarVGGCustomizedArchitecture(NUM_CLASSES, CHANNELS).to(device)
    model_audio.load_state_dict(torch.load(model_path))
    model_audio.eval()
    
    transform = ToTensor()
    test_audio_dataset = CustomSpectrogramDatasetWithPaths(os.path.join(dataset_path, "test"), transform=transform)
    test_audio_loader = DataLoader(test_audio_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    correct = 0
    total = 0
    all_predicted_labels = []
    all_actual_labels = []
    musical_piece_dict = {}
    
    with torch.no_grad():
        for inputs, labels, paths in test_audio_loader:
            inputs = inputs.to(device)
            labels = torch.argmax(labels, dim=1).to(device)
            
            outputs = model_audio(inputs)
            _, predicted = outputs.max(1)
            
            musical_piece_name = list(paths)[0].split("/").pop()[:-4]
            mapped_labels = {0: [1., 0., 0., 0.], 1: [0., 1., 0., 0.], 2: [0., 0., 1., 0.], 3: [0., 0., 0., 1.]}
            musical_piece_dict[musical_piece_name] = [outputs[0].tolist(), mapped_labels[labels[0].tolist()]]
            all_actual_labels.append(predicted.cpu().numpy())
            all_predicted_labels.append(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
                
    return musical_piece_dict
 
 
def validate_lyrics_model(model_path, dataset_path, database_path):
    hyperparameters = {
                            'tokenizer':{
                                'do_lower_case': True,
                                'num_embeddings': 128,
                            },
                            'model':{
                                'num_labels': 4,
                                'batch_size': 1,
                            }
                        }
        
    model = load_model(lyrics_model_path)
    _, test_dataset, _ = load_dataset(dataset_path, database_path)

    tokienizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case = hyperparameters['tokenizer']['do_lower_case'])

    test_labels = np.array(test_dataset['mood'].tolist())
    test_input_ids = tokenize_inputs(hyperparameters, test_dataset['lyric'].tolist(), tokienizer)
    test_attention_masks  = create_attention_masks(test_input_ids)
    test_input_ids, test_attention_masks, test_labels = to_tensor(test_input_ids, test_attention_masks, test_labels)
    test_dataset, test_dataloader = to_custom_tensorDataset_dataLoader_tuple(test_input_ids,
                                                                                test_attention_masks,
                                                                                test_labels, 
                                                                                hyperparameters,
                                                                                test_dataset['mood'].index.values.tolist())
    
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    y_pred = []
    index_result_dict = {}

    with torch.no_grad():
        for index, batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            output = model(b_input_ids, attention_mask=b_input_mask)
            output = output.detach().cpu().numpy()

            for predict in np.argmax(output, axis=1):
                y_pred.append(predict)
            
            index_result_dict[index[0]] = output[0].tolist()

    return index_result_dict
    
    
def create_merged_dict(final_lyric_dict_with_predictions, final_audio_dict_with_predictions):
    merged_dict = {}
    for key in set(final_lyric_dict_with_predictions.keys()).union(final_audio_dict_with_predictions.keys()):
        if key in final_lyric_dict_with_predictions and key in final_audio_dict_with_predictions:
            merged_dict[key] = [(a + b)/2 for a, b in zip(final_lyric_dict_with_predictions[key], 
                                                      final_audio_dict_with_predictions[key])]
        elif key in final_lyric_dict_with_predictions:
            merged_dict[key] = final_lyric_dict_with_predictions[key]
        elif key in final_audio_dict_with_predictions:
            merged_dict[key] = final_audio_dict_with_predictions[key]
    return merged_dict
    
    
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]
    
    
if __name__ == "__main__":
    database_path = "../database/MoodyLyrics4Q_cleaned_split.csv"
    audio_dataset_path = "../database/melgrams/gray/different-params/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray"
    audio_model_path = "./audio/trained_models/torch/checkpoints10/sarkar_59.20_317.pth"
    lyrics_dataset_path = "../database/lyrics"
    lyrics_model_path = "./lyric/xlnet/xlnet_2023-09-02_15-42-29.pt"
    
    name = "to_be_defined"
    label_names = ["happy", "angry", "sad", "relaxed"]
    IM_WIDTH = 1292
    IM_HEIGHT = 128
    
    #this dict contains name of tested musical piece e.g. ML391 with list of lists predicted values and true values 
    #e.g. [[0.23232, 0.32131, 0.32131, 0.2121], [0., 0., 0., 1.]]
    #example record: 'ML792': [[0.15000547468662262, 1.1985995769500732, 0.8381218314170837, -3.4543306827545166], [0.0, 0.0, 0.0, 1.0]]}
    #with the soft voting method it will be sufficient to add values predicted by me and you and take the highest one
    #as the final model's prediction
    audio_piece_dict = validate_audio_model(model_path=audio_model_path, dataset_path=audio_dataset_path, img_width=IM_WIDTH, 
                   img_height=IM_HEIGHT, label_names=label_names, metrics_file=f"{name}.txt", 
                   confusion_matrix_prefix=name)
    final_lyric_dict_with_predictions = validate_lyrics_model(model_path=lyrics_model_path, dataset_path=lyrics_dataset_path,
                    database_path=database_path)
    
    #merge dicts
    final_audio_dict_with_predictions = {key: value[0] for key, value in audio_piece_dict.items()}
    final_dict_with_true_labels = {key: value[1] for key, value in audio_piece_dict.items()}  
    merged_dict = create_merged_dict(final_audio_dict_with_predictions, final_lyric_dict_with_predictions)
    dict_with_int_results = {}
    dict_with_true_results = {}
    
    #find indexes of given results
    for key in merged_dict:
        max_id = argmax(merged_dict[key])
        dict_with_int_results[key] = max_id

    for key in final_dict_with_true_labels:
        max_id = np.argmax(final_dict_with_true_labels[key])
        dict_with_true_results[key] =  max_id 
    
    #sort dicts and create lists to do CM and calculate metrics
    sorted_true_values_list_of_tuples = sorted(dict_with_true_results.items(), key=operator.itemgetter(0))
    sorted_predicted_values_list_of_tuples = sorted(dict_with_int_results.items(), key=operator.itemgetter(0))
    list_of_true_indexes = []
    list_of_predicted_indexes = []
    
    for tuple1_, tuple2_ in zip(sorted_true_values_list_of_tuples, sorted_predicted_values_list_of_tuples):
        list_of_true_indexes.append(tuple1_[1])
        list_of_predicted_indexes.append(tuple2_[1])
    
    assert(len(list_of_true_indexes) == len(list_of_predicted_indexes)), "Error Message: Dicts length are different"
    
    cfm = confusion_matrix(list_of_true_indexes, list_of_predicted_indexes)
    print(cfm)
