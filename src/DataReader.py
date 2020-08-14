import os
import torch
import json
from tqdm import tqdm
import numpy as np
from nltk import tokenize


def readDataFromJson(filename, l2i = None, i2l = None, language = 'english', part = 1.0):
    '''
    使用NLTK的 sent_tokenize 分割句子，丢失newLine，存在误差
    part: used to only preprocess part of data
    '''
    if not filename.endswith('.json'):  return None

    docData = []
    # normalized labels for whole docs with the same type
    if l2i == None and i2l == None:
        label2idx = {'<PAD>':0}
        idx2label = {0:'<PAD>'}   
    else:
        label2idx = l2i
        idx2label = i2l

    with open(filename, 'r') as f:
        rawData = json.load(f)                         # list of dict
        lens = len(rawData)
        if not (part == 1.0):
            lens = int(lens * part)

        print('\n####Reading data from JSON file####')
        for doc in tqdm(rawData[:lens]):
            text = doc['text']
            annotations = doc['annotations']        # list of dict

            ## deal with annotations
            for sec in annotations:
                # begin = sec['begin']
                # length = sec['length']
                # secLabels[begin + length] = sec['sectionLabel']
                if sec['sectionLabel'] not in label2idx:
                    idx = len(label2idx)
                    label2idx[sec['sectionLabel']] = idx 
                    idx2label[idx] = sec['sectionLabel']

            # annotate sentences
            sens = []
            labels = []
            totalLen = 0
            # sentences = text.split('.')
            sentences = tokenize.sent_tokenize(text, language = language)
            for s in sentences:
                totalLen += len(s)
                sens.append(s)
                # find the label
                for sec in annotations:
                    if totalLen <= sec['begin'] + sec['length']:
                        labels.append(label2idx[sec['sectionLabel']])
                        break

            docData.append([sens, labels])
    
    return docData, label2idx, idx2label

def readDataFromEmbedding(data_dir):

    X = []
    Y = []
    label2idx = {}
    idx2label = {}

    print('\n####Reading data from embedding vectors####')
    files = sorted(os.listdir(data_dir))

    '''make sure files[-1] is the dict file'''
    for i in tqdm(range(len(files[:-1]))):
        if i % 2 == 0:
            xpath = os.path.join(data_dir, files[i])
            ypath = os.path.join(data_dir, files[i+1])
            x = torch.from_numpy(np.load(xpath))
            y = torch.from_numpy(np.load(ypath))
            X.append(x)
            Y.append(y)

    # read dict
    i2l_file = os.path.join(data_dir, 'dict', 'idx2label.json')
    l2i_file = os.path.join(data_dir, 'dict', 'label2idx.json')

    with open(i2l_file) as f:
        idx2label = json.load(f)
    with open(l2i_file) as f:
        label2idx = json.load(f)  

    data = {'text': X, 'label':Y}

    return data, label2idx, idx2label

if __name__ == '__main__':
    filename = '/Users/liujun/myProject/Edin-course/thesis/dataset/wikisection_dataset_json/wikisection_en_city_test.json'
    dict_dir = ''
    doc, label2idx, idx2label = readDataFromJson(filename)
    print(len(doc))


