import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set(style="ticks", color_codes=True)
from DataReader import readDataFromJson, readDataFromEmbedding
# import sys
# sys.path.append('.')

hyper_config = {
    'dataset'          : 'disease',
}
config = {
    'dict'             : './sbert_en_disease_train/dict',   
    'train_embed_dir'  : 'sbert_en_disease_train',
    'dev_embed_dir'    : 'sbert_en_disease_dev',

    'use_tfidf'        : False,
    'sen_emd_model'    : 'naive-bert',
    'logging_path'     : './logging/naiveBERT_biLSTM_en_{}_model.log'.format(hyper_config['dataset']),
    'modelName'        : '../models/naiveBERT_biLSTM_en_{}_model.pt'.format(hyper_config['dataset']),    
    'train_json_file'  : '../dataset/WikiSection/wikisection_en_{}_train.json'.format(hyper_config['dataset']),
    'dev_json_file'    : '../dataset/WikiSection/wikisection_en_{}_validation.json'.format(hyper_config['dataset']),
    'test_json_file'   : '../dataset/WikiSection/wikisection_en_{}_test.json'.format(hyper_config['dataset'])
}

train_doc, label2idx, idx2label = readDataFromJson(config['train_json_file'], part = 1.0)
dev_doc, label2idx, idx2label   = readDataFromJson(config['dev_json_file'], i2l= idx2label, l2i=label2idx, part = 1.0)
test_doc, label2idx, idx2label   = readDataFromJson(config['test_json_file'], i2l= idx2label, l2i=label2idx, part = 1.0)


numLabels = []
Labels = []
source = []

for _,label in train_doc:
    numLabels.extend(label)
    Labels.extend([idx2label[idx] for idx in label])
    source.extend(['Train'] * len(label))

for _,label in dev_doc:
    numLabels.extend(label)
    Labels.extend([idx2label[idx] for idx in label])
    source.extend(['Dev'] * len(label))

for _,label in test_doc:
    numLabels.extend(label)
    Labels.extend([idx2label[idx] for idx in label])
    source.extend(['Test'] * len(label))

df = {'numLabels':numLabels, 'Labels':Labels, 'source':source}
data = pd.DataFrame(df)

plot = sns.catplot(y="Labels", hue="source", kind="count",
            order=data.Labels.value_counts().index,
            palette="pastel", edgecolor=".6",data=data)
plot.set(xlabel='', ylabel='Normalized Topic Labels')
plot.savefig('../pic/disease_dist.pdf')  # pdf format is more clear
print('Finished')