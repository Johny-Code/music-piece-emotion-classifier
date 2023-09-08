import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
sys.path += ["lyric/implementation/", "audio"]

from train_sarkar_torch import SarkarVGGCustomizedArchitecture
from CustomXLNetForMultiLabelSequenceClassification import CustomXLNetForMultiLabelSequenceClassification


class EnsembleModel(nn.Module):
    def __init__(self, audioModel: SarkarVGGCustomizedArchitecture, lyricsModel: CustomXLNetForMultiLabelSequenceClassification,
                 nb_classes, batch_size):
        super(EnsembleModel, self).__init__()
        self.audioModel = audioModel
        self.lyric_output_size = 16
        self.audio_output_size = 256
        self.expected_size = (batch_size, self.lyric_output_size)
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
        self.Linear1 = nn.Linear(self.lyric_output_size+self.audio_output_size, 512)
        self.ReLU1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.Linear2 = nn.Linear(512, self.nb_classes)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, audio_x, input_ids, attention_mask, labels):
        x1 = self.truncated_audio_model(audio_x)
        x2 = self.truncated_lyrics_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=None)
        
        # if x2.size(0) < self.expected_size[0]:
        #     padding_size = self.expected_size[0] - x2.size(0)
        #     padding = torch.zeros((padding_size, self.expected_size[1]), device=x2.device)
        #     x2 = torch.cat((x2, padding), dim=0)
        #     print(f"x2.size after padding {x2.size()}")   

        x = torch.cat((x1,x2), dim=1)
        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = self.ReLU1(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.output(x)
        return x
