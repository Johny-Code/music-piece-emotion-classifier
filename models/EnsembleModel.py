import torch
import torch.nn as nn
from torchsummary import summary


class EnsembleModel(nn.Module):
    def __init__(self, audioModel, lyricsModel, nb_classes):
        super(EnsembleModel, self).__init__()
        self.audioModel = audioModel
        self.lyricsModel = lyricsModel
        self.nb_classes = nb_classes
        
        self.truncated_audio_model = torch.nn.Sequential(*list(self.audioModel.children())[:-5])
                
        #to be corrected, maybe the layer with 768-4 neurons should be changed to e.g. 768-256
        self.truncated_lyrics_model = torch.nn.Sequential(*list(self.lyricsModel.children())[:-5])
        
        #freeze already trained layers
        for (name_audio, param_audio), (name_lyrics, param_lyrics) in zip(self.truncated_audio_model.named_parameters(), self.truncated_lyrics_model.named_parameters()):
            if param_audio.requires_grad:
                param_audio.requires_grad = False
            if param_lyrics.requires_grad:
                param_lyrics.requires_grad = False
    
        self.Flatten1 = nn.Flatten()        
        self.Linear1 = nn.Linear(512, 256)
        self.ReLU1 = nn.ReLU()
        self.Linear2 = nn.Linear(256, self.nb_classes)
        self.output = nn.Softmax(dim=1)
        
        
    def forward(self, x1, x2):
        x1 = self.truncated_audio_model(x1)
        x2 = self.truncated_lyrics_model(x2)
        x = torch.cat((x1,x2), dim=1)
        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = self.ReLU1(x)
        x = self.Linear2(x)
        x = self.output(x)
        return x
