import torch
import torch.nn as nn
from transformers import XLNetModel


class CustomXLNetForMultiLabelSequenceClassification(torch.nn.Module):
    
    def __init__(self):
        super(CustomXLNetForMultiLabelSequenceClassification, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.xlnet = self.xlnet.base_model
        self.classifier = torch.nn.Linear(768, 64)
        torch.nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None):
        last_hidden_state = self.xlnet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        mean_last_hidden_state = torch.mean(last_hidden_state[0], 1)
        output = self.classifier(mean_last_hidden_state)
        return output
