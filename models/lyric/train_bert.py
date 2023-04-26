import torch
from transformers import BertTokenizer, BertModel, BertConfig

def define_BERT_model(model_name):
    
    model = BertModel.from_pretrained(model_name, output_hidden_states = True)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    return model, tokenizer

if __name__ == '__main__':
    model_name = 'bert-large-cased'

    model, tokenizer = define_BERT_model(model_name)
    print(f'model config: {model.config}')
    print(f'tokenizer: {tokenizer}')