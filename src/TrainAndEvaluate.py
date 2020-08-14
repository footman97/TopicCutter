import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import logging
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score, classification_report, confusion_matrix
from TopicSeg import TopicSegModel


''' prepare the data set'''
class MyData(Dataset):
    def __init__(self, data_seq):
        self.data_dict = data_seq

    def __len__(self):
        return len(self.data_dict['text'])

    def __getitem__(self, idx):
        text = self.data_dict['text'][idx]
        label = self.data_dict['label'][idx]
        sample = {"text": text, "label": label}
        return sample

def collate_fn(data):

    data.sort(key = lambda x: len(x['text']), reverse = True)
    data_length = [len(sq['text']) for sq in data]
    text  = [d['text'] for d in data]
    label = [d['label'] for d in data]
    text = rnn_utils.pad_sequence(text, 
                                  batch_first = True, padding_value = 0)
    label = rnn_utils.pad_sequence(label, 
                                  batch_first = True, padding_value = 0)
    return {'text': text, 'label': label}, data_length

def loadData(train_data = None, dev_data = None, test_data = None, BATCH_SIZE = 5):

    if train_data:
        train = MyData(train_data)
        train_iter = DataLoader(train, batch_size = BATCH_SIZE, 
                            shuffle = True, collate_fn = collate_fn)
    else:
        train_iter = None
    if dev_data:
        dev   = MyData(dev_data)
        dev_iter   = DataLoader(dev, batch_size = BATCH_SIZE, 
                            shuffle = True, collate_fn = collate_fn)    
    else:
        dev_iter = None
    if test_data:
        test  = MyData(test_data)     
        test_iter  = DataLoader(test, batch_size = BATCH_SIZE, 
                            shuffle = True, collate_fn = collate_fn)  
    else:
        test_iter = None

    return train_iter, dev_iter, test_iter

''' evaluation metrics'''
def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    In the multi-label classification problem, acc equals micro f1 score
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

def cal_acc_eachClass(y_pred, y_true, tag_pad_idx, idx2label):
    '''calculate the accuracy for each class'''
    # y_true = y_true_.detach().numpy()
    matrix = confusion_matrix(y_true, y_pred)
    acc_each = matrix.diagonal()/matrix.sum(axis=1)
    acc_each = np.around(acc_each, decimals=2)
    i2l = idx2label
    for k,v in i2l.items():
        i2l[k] = v.split('.')[-1]
    try:
        acc_each = [[i2l[i], acc_each[i]] for i in range(len(acc_each)) if i != tag_pad_idx]
    except:
        acc_each = [[i2l[str(i)], acc_each[i]] for i in range(len(acc_each)) if i != tag_pad_idx]
    acc_each = sorted(acc_each, key=lambda x:x[1], reverse=True)
    acc_each = list(zip(*acc_each))
    df = pd.DataFrame({'Labels': acc_each[0], 'Accuracy':acc_each[1]})
    return df

def cal_f1_score(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim = 1, keepdim = True)
    non_pad_elements = (y != tag_pad_idx).nonzero()
    p_y = max_preds[non_pad_elements].squeeze(1).detach().numpy()
    l_y = y[non_pad_elements].detach().numpy()
    f1 = f1_score(l_y, p_y, average = 'micro')
    return f1

def cal_mAP(preds, y, tag_pad_idx):

    CLASS = preds.size(-1) - 1 # ignore PADDING index
    AP = []
    for c in range(1, CLASS):
        if torch.cuda.is_available():
            c_y_pred = preds[:,c].cpu().detach().numpy()
            c_y_labl = (y == c).int().cpu().detach().numpy()
        else:
            c_y_pred = preds[:,c].detach().numpy()
            c_y_labl = (y == c).int().detach().numpy()
        ap = average_precision_score(c_y_labl, c_y_pred, average = 'micro')
        if not np.isnan(ap):
            AP.append(ap)
    return np.mean(AP)

'''helper functions'''
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
def init_weights(m):
    '''initialize the model parameters'''
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)

def count_parameters(model):
    '''parameters stat'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


''' train and evaluation'''
def trainModel(topic_seg_model, iterator, optimizer, criterion, PADDING_INDEX = 0):

    topic_seg_model.train()

    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_mAP = 0.0

    hasGPU = torch.cuda.is_available()

    for data, batch_x_len in tqdm(iterator):

        batch_x = data['text']
        batch_label = data['label'] 

        if hasGPU:
            batch_x = batch_x.cuda()
            batch_label = batch_label.cuda()

        batch_x_pack = rnn_utils.pack_padded_sequence(batch_x,
                                                batch_x_len, batch_first = True)

        optimizer.zero_grad()

        batch_y, out1 = topic_seg_model(batch_x_pack)

        # reshape for crossEntropyLoss
        batch_y = batch_y.view(-1, batch_y.shape[-1])
        batch_label = batch_label.view(-1)
        loss = criterion(batch_y, batch_label)
        acc = categorical_accuracy(batch_y, batch_label, PADDING_INDEX)
        mAP = cal_mAP(batch_y, batch_label, PADDING_INDEX)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_mAP += mAP.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_mAP / len(iterator)

def evaluationModel(topic_seg_model, iterator, criterion, PADDING_INDEX = 0, accDict={}):

    epoch_loss = 0
    epoch_acc = 0
    epoch_mAP = 0.0
    
    topic_seg_model.eval()
    hasGPU = torch.cuda.is_available()

    y_true = []
    y_pred = []
    pred_res = {}
    idx = 0
    
    with torch.no_grad():
    
        for data, batch_x_len in tqdm(iterator):

            batch_x = data['text']
            batch_label = data['label'] 

            if hasGPU:
                batch_x = batch_x.cuda()
                batch_label = batch_label.cuda()
            # 
            batch_x_pack = rnn_utils.pack_padded_sequence(batch_x,
                                                    batch_x_len, batch_first = True)

            batch_y, out1 = topic_seg_model(batch_x_pack)

            # reshape for crossEntropyLoss
            batch_y = batch_y.view(-1, batch_y.shape[-1])
            batch_label = batch_label.view(-1)

            
            y_true.append(batch_label)
            y_pred.append(batch_y.argmax(dim = 1, keepdim = True))

            pred_res[idx] = (batch_label, batch_y.argmax(dim = 1, keepdim = True).tolist())
            idx += 1

            loss = criterion(batch_y, batch_label)
            acc = categorical_accuracy(batch_y, batch_label, PADDING_INDEX)
            mAP = cal_mAP(batch_y, batch_label, PADDING_INDEX)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_mAP += mAP.item()
    
    accDict['y_true'] = torch.cat(y_true).tolist()
    accDict['y_pred'] = torch.cat(y_pred).tolist()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_mAP / len(iterator), pred_res


