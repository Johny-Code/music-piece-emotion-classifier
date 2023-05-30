import argparse
import sys
import os
import torch
import numpy as np

from transformers import XLNetTokenizer, XLNetForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from sklearn.metrics import classification_report

from train_svm import SEED

sys.path.append('tools/')
from extract_features_from_lyric import load_en_dataset, clean_lyric


def preprocess(dataset, remove_newline):

    target_dict = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}

    for index, row in dataset.iterrows():
        lyric, _ = clean_lyric(row['lyric'], row['title'])

        if remove_newline:
            lyric = lyric.replace('\n', ' ')
        
        dataset.at[index, 'lyric'] = lyric
        dataset.at[index, 'target'] = target_dict[dataset.at[index, 'mood']]

    dataset = dataset[['mood', 'lyric']]

    print(dataset.head())

    return dataset

def tokenize_lyric(texts, hyperparameters):

    lyrics = texts.to_list() 

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    full_input_ids = []
    full_input_masks = []
    full_segment_ids = []

    SEG_ID_A   = 0
    SEG_ID_CLS = 2
    SEG_ID_PAD = 4

    CLS_ID = tokenizer.encode("<cls>")[0]
    SEP_ID = tokenizer.encode("<sep>")[0]

    for i, lyric in enumerate(lyrics):

        tokens_a = tokenizer.encode(lyric)

        # trim the len of text to max_len
        if len(tokens_a) > hyperparameters['max_seq_length'] - 2:
            print(f'lyric {i} is too long, len = {len(tokens_a)}')
            tokens_a = tokens_a[:hyperparameters['max_seq_length'] - 2]

        tokens = []
        segment_ids = []

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(SEG_ID_A)

        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_A)

        tokens.append(CLS_ID)
        segment_ids.append(SEG_ID_CLS)

        input_ids = tokens

        # The mask has 0 for real tokens and 1 for padding tokens. Only real tokens are attended to.
        input_mask = [0] * len(input_ids)
        
        # Zero-pad up to the sequence length at fornt
        if len(input_ids) < hyperparameters['max_seq_length']:
            delta_len = hyperparameters['max_seq_length'] - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

        assert len(input_ids) == hyperparameters['max_seq_length']
        assert len(input_mask) == hyperparameters['max_seq_length']
        assert len(segment_ids) == hyperparameters['max_seq_length']
        
        full_input_ids.append(input_ids)
        full_input_masks.append(input_mask)
        full_segment_ids.append(segment_ids)

    return full_input_ids, full_input_masks, full_segment_ids

def vec_to_tensor(inputs, tags, masks, segs):

    if len(inputs) != len(tags) != len(masks) != len(segs):
        print("Training vactors haven't got same length")
        exit(0)
        
    inputs = torch.tensor(inputs)
    tags = torch.tensor(tags)
    masks = torch.tensor(masks)
    segs = torch.tensor(segs)

    return inputs, tags, masks, segs


def fine_tune(tr_inputs, tr_tags, tr_masks, tr_segs, val_inputs, val_tags, val_masks, val_segs, hyperparameters):

    print('Start fine-tuning...')
    print(tr_tags[:10])

    tr_inputs, tr_tags, tr_masks, tr_segs = vec_to_tensor(tr_inputs, tr_tags, tr_masks, tr_segs)
    val_inputs, val_tags, val_masks, val_segs = vec_to_tensor(val_inputs, val_tags, val_masks, val_segs)

    train_data = TensorDataset(tr_inputs, tr_masks,tr_segs, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=hyperparameters['batch_size'], drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    num_labels = 4  

    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=num_labels, problem_type="multi_label_classification")
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
            'weight_decay_rate': hyperparameters['weight_decay']
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay_rate': 0.0
        }
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=hyperparameters['lr'], eps=hyperparameters['eps'])

    model.train()
    print("***** Running training *****")
    print(f"  Num examples = {len(tr_inputs)}")
    print(f"  Batch size = {hyperparameters['batch_size']}")
    print(f"  Num steps = {hyperparameters['epochs']}")


    for _ in range(hyperparameters['epochs']):
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_segs,b_labels = batch

            # forward pass
            outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
            loss, logits = outputs[:2]
            if n_gpu>1:
                loss = loss.mean()

            # backward pass
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=hyperparameters['max_grad_norm'])

            # update parameters
            optimizer.step()
            optimizer.zero_grad()

        print(f"Train loss: {tr_loss/nb_tr_steps}")

    return model

def test_model(model, test_inputs, test_tags, test_masks, test_segs):

    test_data = TensorDataset(test_inputs, test_masks, test_segs, test_tags)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=hyperparameters['batch_size'], drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    eval_loss = 0
    eval_accuracy = 0
    nb_eval_steps = 0
    
    y_true = []
    y_pred = []

    print("***** Running evaluation *****")
    print(f"  Num examples = {len(test_inputs)}")
    print(f"  Batch size = {hyperparameters['batch_size']}")

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_segs, b_labels = batch

        with torch.no_grad():
            outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
            tmp_eval_loss, logits = outputs[:2]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        for predict in np.argmax(logits, axis=1):
            y_pred.append(predict)

        for true in label_ids.tolist():
            y_true.append(true)

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    
    eval_loss = eval_loss / nb_eval_steps

    print(classification_report(y_true, y_pred))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--simple_run', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    args = parser.parse_args()

    if args.simple_run:
        
        dataset_path = os.path.join('..', 'database', 'lyrics')
        duplicated_path = os.path.join('database', 'removed_rows.json') 

        en_dataset = load_en_dataset(dataset_path, duplicated_path)

        remove_newline = True
        dataset = preprocess(en_dataset, remove_newline)

        hyperparameters = {'batch_size': 32,
                           'epochs': 10,
                            'lr': 2e-5, 
                            'eps': 1e-8, 
                            'max_grad_norm': 1.0, 
                            'warmup_steps': 0, 
                            'weight_decay': 0.0,
                            'max_grad_norm': 1.0,
                            'max_seq_length': 32,
                            }

        print('hyperparameters:')
        for key, value in hyperparameters.items():
            print(key, ' : ', value)

        
        full_input_ids, full_input_masks, full_segment_ids = tokenize_lyric(dataset['lyric'], hyperparameters)
        tags = dataset['mood'].to_list()
        
        tr_inputs, test_inputs, tr_tags, test_tags, tr_masks, test_masks, tr_segs, test_segs = train_test_split(full_input_ids, tags, full_input_masks, full_segment_ids, random_state=SEED, test_size=0.3)
        
        val_inputs, test_inputs, val_tags, test_tags, val_masks, test_masks, val_segs, test_segs = train_test_split(test_inputs, test_tags, test_masks, test_segs, random_state=SEED, test_size=0.5)    
        
        model = fine_tune(tr_inputs, tr_tags, tr_masks, tr_segs, val_inputs, val_tags, val_masks, val_segs, hyperparameters)
        
        test_model(model, test_inputs, test_tags, test_masks, test_segs)

    elif args.grid_search:
        pass

    else:
        print('Please specify --simple_run or --grid_search')
        print('For simple run: python train_svm.py --simple_run')
        print('For grid search: python train_svm.py --grid_search')
        sys.exit(0)