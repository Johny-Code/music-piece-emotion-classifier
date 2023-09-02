import torch
from transformers import XLNetModel


class CustomXLNetForMultiLabelSequenceClassification(torch.nn.Module):
    
    def __init__(self):
        super(CustomXLNetForMultiLabelSequenceClassification, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        
        # Remove the last layer (classifier) from the XLNet model
        self.xlnet = self.xlnet.base_model
    
        # Create a custom classifier layer
        self.classifier = torch.nn.Linear(768, 256)  # Adjust the output dimension as needed
        torch.nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None):
        last_hidden_state = self.xlnet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # last hidden layer
        mean_last_hidden_state = torch.mean(last_hidden_state[0], 1)  # Pool the outputs into a mean vector
        output = self.classifier(mean_last_hidden_state)
        return output
