import os
import sys
import operator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
sys.path += ["../utils/", "./audio/implementation", "./audio", "./lyric"]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from contextlib import redirect_stdout
from sklearn.metrics import classification_report, confusion_matrix
from draw_plot import plot_acc_loss_torch, draw_confusion_matrix
from CustomSpectrogramDatasetWithPaths import CustomSpectrogramDatasetWithPaths
from train_sarkar_torch import SarkarVGGCustomizedArchitecture


def validate_audio_model(model_path, dataset_path, img_height, img_width, label_names, metrics_file, 
                         confusion_matrix_prefix):
    NUM_CLASSES = 4
    CHANNELS = 1
    BATCH_SIZE = 1
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
    audio_dataset_path = "../database/melgrams/gray/different-params/melgrams_2048_nfft_1024_hop_128_mel_jpg_proper_gray"
    audio_model_path = "./audio/trained_models/torch/checkpoints3/sarkar_45.48.pth"
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

    ####to be implemented, now just example is here, dict must be implemented in the given below format 
    final_lyric_dict_with_predictions = {'ML792': [0.5104888081550598, 0.12528406083583832, 0.2753966748714447, 0.08883053809404373]}
    ####
    
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
