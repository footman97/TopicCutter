import os
import time
import sys
import numpy as np
import logging
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set(style="ticks", color_codes=True)
from SentenceEmbedding import docsEmbedding
from DataReader import readDataFromJson, readDataFromEmbedding
from TrainAndEvaluate import loadData, trainModel, evaluationModel
from TrainAndEvaluate import epoch_time, init_weights, count_parameters, cal_acc_eachClass
from TopicSeg import TopicSegModel


hyper_config = {
    'dataset'          : 'disease',
}
config = {
    'dict'             : './sbert_en_disease_train/dict',   
    'train_embed_dir'  : 'sbert_en_disease_train',
    'dev_embed_dir'    : 'sbert_en_disease_dev',

    'use_tfidf'        : False,
    'sen_emd_model'    : 'sbert',
    'logging_path'     : './logging/sbert_biLSTM_en_{}_model.log'.format(hyper_config['dataset']),
    'modelName'        : '../models/sbert_biLSTM_en_{}_model.pt'.format(hyper_config['dataset']),    
    'train_json_file'  : '../dataset/WikiSection/wikisection_en_{}_train.json'.format(hyper_config['dataset']),
    'dev_json_file'    : '../dataset/WikiSection/wikisection_en_{}_validation.json'.format(hyper_config['dataset']),
    'test_json_file'   : '../dataset/WikiSection/wikisection_en_{}_test.json'.format(hyper_config['dataset'])
}


train_doc, label2idx, idx2label = readDataFromJson(config['train_json_file'], part = 1.0)
dev_doc, label2idx, idx2label   = readDataFromJson(config['dev_json_file'], i2l= idx2label, l2i=label2idx, part = 1.0)
test_doc, label2idx, idx2label   = readDataFromJson(config['test_json_file'], i2l= idx2label, l2i=label2idx, part = 0.02)
test_data = docsEmbedding(test_doc, modelFlag = config['sen_emd_model'], tf_idf_weight = config['use_tfidf'])

N_EPOCHS = 20
PADDING_INDEX = 0
EMBEDDING_DIM = test_data['text'][0].size(-1)
HIDDEN_DIM = 128
OUTPUT_DIM = len(label2idx)
N_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.25
BATCH_SIZE = 5 if config['train_json_file'].find('disease') > 0 else 10

topic_seg_model = TopicSegModel(embedding_dim = EMBEDDING_DIM,
                                hidden_dim = HIDDEN_DIM,
                                output_dim = OUTPUT_DIM,
                                n_layers = N_LAYERS,
                                bidirectional = BIDIRECTIONAL,
                                dropout = DROPOUT)
                                
# prepare the data 
train_iter, dev_iter, test_iter = loadData(None, None, test_data, BATCH_SIZE) 

criterion = nn.CrossEntropyLoss(ignore_index = PADDING_INDEX)

accDict = {}
# report the test result on the best model of validation set
topic_seg_model.load_state_dict(torch.load(config['modelName']))
# train on the GPU
if torch.cuda.is_available():
    topic_seg_model = topic_seg_model.cuda()
test_loss, test_acc, test_mAP = evaluationModel(topic_seg_model, test_iter, criterion, PADDING_INDEX=PADDING_INDEX, accDict=accDict)
logging.info('*' * 15 + 'Test Set Result' + '*' * 15)
print(f'\t test.  Loss: {test_loss:.3f} | Test.  micro-f1: {test_acc*100:.2f}% | Val.  mAP: {test_mAP*100:.2f}%')
logging.info(f'\tVal.  Loss: {test_loss:.3f} | Val.  micro-f1: {test_acc*100:.2f}% | Val.  mAP: {test_mAP*100:.2f}%')
logging.info('*' * 15 + '***************' + '*' * 15)
# plot the accuracy for each topic label
data_df = cal_acc_eachClass(accDict['y_pred'], accDict['y_true'], PADDING_INDEX, idx2label)
modelName = config['modelName'].split('/')[-1]
data_df.to_csv(modelName + 'category.csv')