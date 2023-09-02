import torch
from transformers import XLNetModel


class XLNetForMultiLabelSequenceClassification(torch.nn.Module):
    
    def __init__(self, num_labels=4):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)
        torch.nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None):
        last_hidden_state = self.xlnet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) #last hidden layer
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state) #pool the outputs into a mean vector
        output = self.classifier(mean_last_hidden_state)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(output, labels.float())
            return loss
        else:
            probs = torch.softmax(output, dim=1)
            return probs
        
    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True
    
    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector 
        """
        last_hidden_state = last_hidden_state[0]
        # TODO it can be done in different ways, not only mean
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state
