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
    'dataset'          : 'city',
}
config = {
    'dict'             : './sbert_en_disease_train/dict',   
    'train_embed_dir'  : 'sbert_en_disease_train',
    'dev_embed_dir'    : 'sbert_en_disease_dev',

    'use_tfidf'        : False,
    'sen_emd_model'    : 'naive-bert',
    'logging_path'     : './logging/naiveBERT_layer3_en_{}_model.log'.format(hyper_config['dataset']),
    'modelName'        : '../models/naiveBERT_layer3_en_{}_model.pt'.format(hyper_config['dataset']),    
    'train_json_file'  : '../dataset/WikiSection/wikisection_en_{}_train.json'.format(hyper_config['dataset']),
    'dev_json_file'    : '../dataset/WikiSection/wikisection_en_{}_validation.json'.format(hyper_config['dataset']),
    'test_json_file'   : '../dataset/WikiSection/wikisection_en_{}_test.json'.format(hyper_config['dataset'])
}

logging.basicConfig(filename = config['logging_path'],
                    format = '%(asctime)s - %(message)s',
                    level = logging.INFO)

'''read data from embedding results'''
# train_data, label2idx, idx2label = readDataFromEmbedding(config['train_embed_dir'])
# dev_data, label2idx, idx2label = readDataFromEmbedding(config['dev_embed_dir'])
# test_data = dev_data
'''read data from raw json file'''
train_doc, label2idx, idx2label = readDataFromJson(config['train_json_file'], part = 1.0)
dev_doc, label2idx, idx2label   = readDataFromJson(config['dev_json_file'], i2l= idx2label, l2i=label2idx, part = 1.0)
test_doc, label2idx, idx2label   = readDataFromJson(config['test_json_file'], i2l= idx2label, l2i=label2idx, part = 1.0)
train_data = docsEmbedding(train_doc, modelFlag = config['sen_emd_model'], tf_idf_weight = config['use_tfidf'])
dev_data = docsEmbedding(dev_doc, modelFlag = config['sen_emd_model'], tf_idf_weight = config['use_tfidf'])
test_data = docsEmbedding(test_doc, modelFlag = config['sen_emd_model'], tf_idf_weight = config['use_tfidf'])

N_EPOCHS = 15
PADDING_INDEX = 0
EMBEDDING_DIM = train_data['text'][0].size(-1)
HIDDEN_DIM = 128
OUTPUT_DIM = len(label2idx)
N_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.25
BATCH_SIZE = 5 if config['train_json_file'].find('disease') > 0 else 10

logging.info('*' * 15 + 'model Info' + '*' * 15)
logging.info('model name: {}'.format(config['modelName']))
logging.info('epoches: {}'.format(N_EPOCHS))
logging.info('hidden dimensions: {}'.format(HIDDEN_DIM))
logging.info('layers: {}'.format(N_LAYERS))
logging.info('dropout: {}'.format(DROPOUT))
logging.info('batch size: {}'.format(BATCH_SIZE))
logging.info('*' * 15 + '**********' + '*' * 15)

topic_seg_model = TopicSegModel(embedding_dim = EMBEDDING_DIM,
                                hidden_dim = HIDDEN_DIM,
                                output_dim = OUTPUT_DIM,
                                n_layers = N_LAYERS,
                                bidirectional = BIDIRECTIONAL,
                                dropout = DROPOUT)
# train on the GPU
if torch.cuda.is_available():
    topic_seg_model = topic_seg_model.cuda()

# train the model
topic_seg_model.apply(init_weights)
logging.info(f'The model has {count_parameters(topic_seg_model):,} trainable parameters')
optimizer = optim.Adam(topic_seg_model.parameters())
# criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)
criterion = nn.CrossEntropyLoss(ignore_index = PADDING_INDEX)

# prepare the data 
train_iter, dev_iter, _ = loadData(train_data, dev_data, None, BATCH_SIZE)           
_,_,test_iter           = loadData(None, None, test_data, BATCH_SIZE)                  

best_valid_loss = float('inf')
accDict = {}

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc, train_mAP = trainModel(topic_seg_model, train_iter, optimizer, criterion, PADDING_INDEX)
    valid_loss, valid_acc, valid_mAP,_ = evaluationModel(topic_seg_model, dev_iter, criterion, PADDING_INDEX)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(topic_seg_model.state_dict(), config['modelName'])

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train micro-f1: {train_acc*100:.2f}% | Train mAP: {train_mAP*100:.2f}%')
    print(f'\tVal.  Loss: {valid_loss:.3f} | Val.  micro-f1: {valid_acc*100:.2f}% | Val.  mAP: {valid_mAP*100:.2f}%')

    logging.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    logging.info(f'\tTrain Loss: {train_loss:.3f} | Train micro-f1: {train_acc*100:.2f}% | Train mAP: {train_mAP*100:.2f}%')
    logging.info(f'\tVal.  Loss: {valid_loss:.3f} | Val.  micro-f1: {valid_acc*100:.2f}% | Val.  mAP: {valid_mAP*100:.2f}%')

# report the test result on the best model of validation set
topic_seg_model.load_state_dict(torch.load(config['modelName']))
if torch.cuda.is_available():
    topic_seg_model = topic_seg_model.cuda()
test_loss, test_acc, test_mAP, pred_res = evaluationModel(topic_seg_model, test_iter, criterion, PADDING_INDEX=PADDING_INDEX, accDict=accDict)

for v in pred_res.values():
    logging.info('\n ref: \n')
    logging.info(v[0])
    logging.info('\n' + 'pred: ' +'\n')
    logging.info(v[1]) 

logging.info(idx2label)

logging.info('*' * 15 + 'Test Set Result' + '*' * 15)
print(f'\t test.  Loss: {test_loss:.3f} | Test.  micro-f1: {test_acc*100:.2f}% | Val.  mAP: {test_mAP*100:.2f}%')
logging.info(f'\tVal.  Loss: {test_loss:.3f} | Val.  micro-f1: {test_acc*100:.2f}% | Val.  mAP: {test_mAP*100:.2f}%')
logging.info('*' * 15 + '***************' + '*' * 15)


# plot the accuracy for each topic label
data_df = cal_acc_eachClass(accDict['y_pred'], accDict['y_true'], PADDING_INDEX, idx2label)
modelName = config['modelName'].split('/')[-1]
data_df.to_csv(modelName + 'category.csv')

# plot = sns.lineplot(x="Labels", y="Accuracy", sort=False, data=data_df)
# plot.tick_params(axis='x', labelsize=6)
# plot.set_xlabel('')   # not showing x-axis Labels
# plt.xticks(plt.xticks()[0], data_df.Labels, rotation=30, ha='right')
# plt.yticks(np.arange(0, 1, 0.05))
# plot.figure.tight_layout()
# n = config['modelName'].split('/')[-1].split('.')[0]
# plot.figure.savefig('../pic/{}.pdf'.format(n))