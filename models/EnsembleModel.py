import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
sys.path += ["lyric/implementation/", "audio"]

from train_sarkar_torch import SarkarVGGCustomizedArchitecture
from CustomXLNetForMultiLabelSequenceClassification import CustomXLNetForMultiLabelSequenceClassification


class EnsembleModel(nn.Module):
    def __init__(self, audioModel: SarkarVGGCustomizedArchitecture, lyricsModel: CustomXLNetForMultiLabelSequenceClassification, nb_classes):
        super(EnsembleModel, self).__init__()
        self.audioModel = audioModel
        self.expected_size = (16,256)
        self.lyricsModel = lyricsModel
        self.truncated_lyrics_model = lyricsModel
        self.nb_classes = nb_classes
        self.truncated_audio_model = torch.nn.Sequential(*list(self.audioModel.children())[:-5])

        #freeze already trained layers
        for name_audio, param_audio in self.truncated_audio_model.named_parameters():
            if param_audio.requires_grad:
                param_audio.requires_grad = False
                
        for name_lyrics, param_lyrics in self.truncated_lyrics_model.named_parameters():
            if param_lyrics.requires_grad and not name_lyrics.endswith('classifier.weight') and not name_lyrics.endswith('classifier.bias'):
                param_lyrics.requires_grad = False        
    
        self.Flatten1 = nn.Flatten()        
        self.Linear1 = nn.Linear(512, 256)
        self.ReLU1 = nn.ReLU()
        self.Linear2 = nn.Linear(256, self.nb_classes)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, audio_x, input_ids, attention_mask, labels):
        x1 = self.truncated_audio_model(audio_x)
        x2 = self.truncated_lyrics_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=None)
        
        if x2.size(0) < self.expected_size[0]:
            padding_size = self.expected_size[0] - x2.size(0)
            padding = torch.zeros((padding_size, self.expected_size[1]), device=x2.device)
            x2 = torch.cat((x2, padding), dim=0)
        
        x = torch.cat((x1,x2), dim=1)
        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = self.ReLU1(x)
        x = self.Linear2(x)
        x = self.output(x)
        return x
