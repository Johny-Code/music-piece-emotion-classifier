import argparse
import sys
import os
import torch
import time
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetModel
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from sklearn.metrics import classification_report

from train_svm import SEED

sys.path.append('tools/')
from extract_features_from_lyric import load_en_dataset, clean_lyric

class XLNetForMultiLabelSequenceClassification(torch.nn.Module):
    def __init__(self, num_labels=4):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        
        # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)
        logits = logits[:, 1] - logits[:, 0]

        if labels is not None:
            loss = torch.nn.BCEWithLogitsLoss()(logits, labels.float())

            return loss
        else:
            return logits

    def freeze_xlnet_decoder(self):
    # Freeze XLNet weight parameters. They will not be updated during training.
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):
    # Unfreeze XLNet weight parameters. They will be updated during training.

        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
    # Pool the output vectors into a single mean vector 
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state


def preprocess(dataset, remove_newline):

    # target_dict = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}
    target_dict = {'happy': [1.,0.,0.,0.], 'angry': [0.,1.,0.,0.], 'sad': [0.,0.,1.,0.], 'relaxed': [0.,0.,0.,1.]}


    for index, row in dataset.iterrows():
        lyric, _ = clean_lyric(row['lyric'], row['title'])

        if remove_newline:
            lyric = lyric.replace('\n', ' ')
        
        dataset.at[index, 'lyric'] = lyric
        dataset.at[index, 'mood'] = target_dict[dataset.at[index, 'mood']]

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

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# function to save and load the model form a specific epoch
def save_model(model, save_path, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist):
    """
    Save the model to the path directory provided
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {'epochs': epochs, \
                    'lowest_eval_loss': lowest_eval_loss,\
                    'state_dict': model_to_save.state_dict(),\
                    'train_loss_hist': train_loss_hist,\
                    'valid_loss_hist': valid_loss_hist
                }
    torch.save(checkpoint, save_path)
    print("Saving model at epoch {} with validation loss of {}".format(epochs,\
                                                                        lowest_eval_loss))
    return

def load_model(save_path):
    """
    Load the model from the path directory provided
    """
    checkpoint = torch.load(save_path)
    model_state_dict = checkpoint['state_dict']
    model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])
    model.load_state_dict(model_state_dict)

    epochs = checkpoint["epochs"]
    lowest_eval_loss = checkpoint["lowest_eval_loss"]
    train_loss_hist = checkpoint["train_loss_hist"]
    valid_loss_hist = checkpoint["valid_loss_hist"]

    return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist

def train(model, num_epochs,\
            optimizer,\
            train_dataloader, valid_dataloader,\
            model_save_path,\
            train_loss_set=[], valid_loss_set = [],\
            lowest_eval_loss=None, start_epoch=0,\
            device="cpu"
            ):
    """
    Train the model and save the model with the lowest validation loss
    """
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []
    # Measure the total training time for the whole run.
    total_t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # trange is a tqdm wrapper around the normal python range
    for i in trange(num_epochs, desc="Epoch"):
        # if continue training from saved model
        actual_epoch = start_epoch + i

        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set. 
        print("")
        print('======== Epoch {:} / {:} ========'.format(actual_epoch, num_epochs))
        print('Training...')
        
        # Measure how long the training epoch takes.
        t0 = time.time()
        
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        num_train_samples = 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            print(batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            # store train loss
            tr_loss += loss.item()
            num_train_samples += b_labels.size(0)
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            #scheduler.step()

        # Update tracking variables
        epoch_train_loss = tr_loss/num_train_samples
        train_loss_set.append(epoch_train_loss)

    #     print("Train loss: {}".format(epoch_train_loss))
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(epoch_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()
        
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables 
        eval_loss = 0
        num_eval_samples = 0

        # Evaluate data for one epoch
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate validation loss
                loss = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                # store valid loss
                eval_loss += loss.item()
                num_eval_samples += b_labels.size(0)

        epoch_eval_loss = eval_loss/num_eval_samples
        valid_loss_set.append(epoch_eval_loss)

    #     print("Valid loss: {}".format(epoch_eval_loss))
        
        # Report the final accuracy for this validation run.
    #     avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    #     print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
    #     avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(epoch_eval_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': actual_epoch,
                'Training Loss': epoch_train_loss,
                'Valid. Loss': epoch_eval_loss,
    #             'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        
    if lowest_eval_loss == None:
        lowest_eval_loss = epoch_eval_loss
        # save model
        save_model(model, model_save_path, actual_epoch,\
                    lowest_eval_loss, train_loss_set, valid_loss_set)
    else:
        if epoch_eval_loss < lowest_eval_loss:
            lowest_eval_loss = epoch_eval_loss
            # save model
            save_model(model, model_save_path, actual_epoch,\
                    lowest_eval_loss, train_loss_set, valid_loss_set)
    
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return model, train_loss_set, valid_loss_set, training_stats


def fine_tune(tr_inputs, tr_tags, tr_masks, tr_segs, val_inputs, val_tags, val_masks, val_segs, hyperparameters):

    print('Start fine-tuning...')

    tr_inputs, tr_tags, tr_masks, tr_segs = vec_to_tensor(tr_inputs, tr_tags, tr_masks, tr_segs)
    val_inputs, val_tags, val_masks, val_segs = vec_to_tensor(val_inputs, val_tags, val_masks, val_segs)

    train_data = TensorDataset(tr_inputs, tr_masks,tr_segs, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=hyperparameters['batch_size'], drop_last=True)

    valid_data = TensorDataset(val_inputs, val_masks, val_segs, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=hyperparameters['batch_size'], drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    n_gpu = torch.cuda.device_count()
    print(f'Number of gpu: {n_gpu}')

    model = XLNetForMultiLabelSequenceClassification(num_labels=4)

    optimizer = torch.optim.AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5
                  # eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                 weight_decay=0.01,
                #  correct_bias=False
                )

    num_epochs = hyperparameters['num_epochs']

    model_save_path = hyperparameters['model_save_path']

    model, train_loss_set, valid_loss_set, training_stats = train(model=model,\
                                                              num_epochs=num_epochs,\
                                                              optimizer=optimizer,\
                                                              train_dataloader=train_dataloader,\
                                                              valid_dataloader=valid_dataloader,\
                                                              model_save_path=model_save_path,\
                                                              device="cuda"
                                                              )
    
    return model, train_loss_set, valid_loss_set, training_stats

def test_model(model, test_inputs, test_tags, test_masks, test_segs):

    test_data = TensorDataset(test_inputs, test_masks, test_segs, test_tags)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=hyperparameters['batch_size'], drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    eval_loss = 0
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

def simple_run(hyperparameters):
    
    dataset_path = os.path.join('..', 'database', 'lyrics')
    duplicated_path = os.path.join('database', 'removed_rows.json') 

    en_dataset = load_en_dataset(dataset_path, duplicated_path)

    remove_newline = True
    dataset = preprocess(en_dataset, remove_newline)
    
    full_input_ids, full_input_masks, full_segment_ids = tokenize_lyric(dataset['lyric'], hyperparameters)
    tags = dataset['mood'].to_list()

    tr_inputs, test_inputs, tr_tags, test_tags, tr_masks, test_masks, tr_segs, test_segs = train_test_split(full_input_ids, tags, full_input_masks, full_segment_ids, random_state=SEED, test_size=0.3)
    
    val_inputs, test_inputs, val_tags, test_tags, val_masks, test_masks, val_segs, test_segs = train_test_split(test_inputs, test_tags, test_masks, test_segs, random_state=SEED, test_size=0.5)    
    
    model = fine_tune(tr_inputs, tr_tags, tr_masks, tr_segs, val_inputs, val_tags, val_masks, val_segs, hyperparameters)
    
    test_model(model, test_inputs, test_tags, test_masks, test_segs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--simple_run', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    args = parser.parse_args()

    if args.simple_run:

        output_dir = os.path.join('models', 'lyric', 'xlnet')

        os.makedirs(output_dir, exist_ok=True)
    
        hyperparameters = {'batch_size': 32,
                           'num_epochs': 10,
                            'lr': 2e-5, 
                            'eps': 1e-8, 
                            'max_grad_norm': 1.0, 
                            'warmup_steps': 0, 
                            'weight_decay': 0.0,
                            'max_grad_norm': 1.0,
                            'max_seq_length': 32,
                            'model_save_path': output_dir,
                            }

        print('hyperparameters:')
        for key, value in hyperparameters.items():
            print(key, ' : ', value)

        model, train_loss_set, valid_loss_set, training_stats = simple_run(hyperparameters)

        # Display floats with two decimal places.
        pd.set_option('precision', 2)

        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=training_stats)

        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')

        # Display the table.
        print(df_stats)

        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([0, 1, 2, 3, 4, 5])

        plt.show()

    elif args.grid_search:
        
        print('grid search')

        params = {'batch_size': [16, 32, 64],
                  'epochs': [10, 20, 30],
                  'lr': [1e-5, 1e-4, 0.001, 0.01, 0.1],
                  'max_seq_length': [32, 128, 256, 512, 1024] }            
        
        for batch in params['batch_size']:
            for epoch in params['epochs']:
                for lr in params['lr']:
                    for max_seq_length in params['max_seq_length']:

                        hyperparameters = {'batch_size': batch,
                                           'epochs': epoch,
                                            'lr': lr, 
                                            'eps': 1e-8, 
                                            'max_grad_norm': 1.0, 
                                            'warmup_steps': 0, 
                                            'weight_decay': 0.0,
                                            'max_grad_norm': 1.0,
                                            'max_seq_length': max_seq_length,
                                            }

                        print('hyperparameters:')
                        for key, value in hyperparameters.items():
                            print(key, ' : ', value)

                        simple_run(hyperparameters)

    else:
        print('Please specify --simple_run or --grid_search')
        print('For simple run: python train_svm.py --simple_run')
        print('For grid search: python train_svm.py --grid_search')
        sys.exit(0)