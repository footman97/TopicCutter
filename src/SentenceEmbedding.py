import os
import torch
import numpy as np
import random
from tqdm import tqdm
from DataReader import readDataFromJson
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


def word2vec_embed(sen:str, model, word2weight = None):
    '''use word2vec to get the word embedding, then use the avg pooling to 
    get the sentence embedding, count the jumped word'''
    emd_dim = 300
    sen_emd = []
    words = sen.split(' ')

    if word2weight:
        wordsWeights = [word2weight[w] for w in words]
        norm_weights = [w / sum(wordsWeights) for w in wordsWeights]
        sen_emd = np.sum([model[w] * norm_weights[i] if w in model else np.zeros(emd_dim) for i,w in enumerate(words)],
                        axis=0, keepdims=True, dtype=np.float32)

        # sen_emd = np.mean([model[w] * word2weight[w] if w in model else np.zeros(emd_dim) for w in words], 
        #                   axis=0, keepdims=True, dtype=np.float32)
    else:
        sen_emd = np.mean([model[w] if w in model else np.zeros(emd_dim) for w in words], 
                          axis=0, keepdims=True, dtype=np.float32)
        
    return torch.from_numpy(sen_emd)

def tf_idf_BERT(input_tks, tk_emd, word2weight):

    res = []
    start_idx = 0
    start = ''
    for i, token in enumerate(input_tks):
        if token[:2] == '##':
            start += token[2:]
        else:
            # print(start, start_idx, i)
            for j in range(start_idx, i):
                weight = torch.from_numpy(word2weight[start].astype(np.float32))
                res.append(tk_emd[j] * weight)
            start = token
            start_idx = i
    # print(start, start_idx)
    for j in range(start_idx, len(input_tks)):
        weight = torch.from_numpy(word2weight[start].astype(np.float32))
        res.append(tk_emd[j] * weight)

    res = torch.cat(res).view(-1, 768).sum(dim=0, keepdim=True)   
    return res

def naiveBERT_embed(doc:list, model, tokenizer, word2weight = None, use_CLS = False):
    '''use the naive BERT model to embed the sentence, then to embed the document'''
    doc_emd = []
    hasGPU = True if torch.cuda.is_available() else False
    layers = 9

    for sen in doc:
        '''this model can only deal with 512 pos emd'''
        if len(sen) > 512:  
            sen = sen[:512]
            # print(sen)
            
        input_ids = tokenizer.encode(sen)
        input_tks = tokenizer.tokenize(sen)
        input_ids_t = torch.tensor([input_ids])
        if hasGPU:  input_ids_t = input_ids_t.cuda()

        with torch.no_grad():
            # tk_emd = model(input_ids_t)[0].squeeze(0)
            # output hidden states
            tk_emd = model(input_ids_t)[2][layers]
            if hasGPU:  tk_emd = tk_emd.cpu()

            if word2weight:
                sen_emd = tf_idf_BERT(input_tks, tk_emd, word2weight)
            elif use_CLS:    
                sen_emd = tk_emd[0].unsqueeze(0)
            elif layers:
                layer_hidden_states = tk_emd[:, 1:, :]
                layer_hidden_states = layer_hidden_states.squeeze(0)
                sen_emd = layer_hidden_states.mean(dim = 0, keepdim = True)
            else:        
                sen_emd = tk_emd.mean(dim = 0, keepdim = True)

        doc_emd.append(sen_emd)
    
    return torch.cat(doc_emd)


def docsEmbedding(docData, modelFlag = 'sbert', tf_idf_weight = False, data_dir = None):
    '''
    data_dir        : place to store the vectors data
    '''
    if tf_idf_weight:
        print('####Training the TF-IDF matrix####')
        docs = [' '.join(doc) for doc, _ in docData]
        tfidf = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        tfidf.fit_transform(docs)
        max_idf = max(tfidf.idf_)
        ''' if a word was never seen - it must be at least as infrequent
        as any of the known words - so the default idf is the max of known idf's '''
        word2weight = defaultdict(lambda: max_idf, 
                                  [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    else:
        word2weight = None
    
    if data_dir:
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        print('####Embedding results will be stored at {}####'.format(data_dir))
    else:
        print('####Embedding results will not be stored####')

    X = []
    Y = []
    model, tokenizer = None, None

    print('####Loading the model####')
    if modelFlag == 'sbert':
        # for sentence transformer (SBERT)
        from sentence_transformers import SentenceTransformer
        sentenceModelList = ['bert-base-nli-mean-tokens', 'bert-large-nli-mean-tokens']
        '''No need to convert to GPU here, since the model does for us'''
        model = SentenceTransformer(sentenceModelList[0])

    elif modelFlag == 'naive-bert':
        from transformers import BertModel, BertTokenizer, BertConfig
        # full list https://huggingface.co/transformers/pretrained_models.html
        transformerModelList = [
        'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased'
        ]
        config = BertConfig.from_pretrained(transformerModelList[0], output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(transformerModelList[0])
        model = BertModel.from_pretrained(transformerModelList[0], config=config) 
        if torch.cuda.is_available():
            model = model.cuda()        
    elif modelFlag == 'word2vec':
        import gensim   
        model = gensim.models.KeyedVectors.load_word2vec_format(\
            './word2vec/GoogleNews-vectors-negative300.bin', binary=True) 

    idx = 0
    for doc, docLabels in tqdm(docData):

        if modelFlag == 'sbert':
            docEmbedding = model.encode(doc)
        elif modelFlag == 'naive-bert':
            docEmbedding = naiveBERT_embed(doc, model, tokenizer, word2weight = word2weight, use_CLS = False)
        elif modelFlag == 'word2vec':
            doc_emd_temp = []
            for sen in doc:
                sen_emd = word2vec_embed(sen, model, word2weight = word2weight)
                doc_emd_temp.append(sen_emd)

            docEmbedding = torch.cat(doc_emd_temp)

        '''append the embedding result'''
        X.append(torch.Tensor(docEmbedding))
        Y.append(torch.LongTensor(docLabels))

        if data_dir:
            '''save the embedding result'''
            x_numpy = np.array(docEmbedding)
            y_numpy = np.array(docLabels)
            np.save(os.path.join(data_dir, str(idx) + '_x.npy'), x_numpy)
            np.save(os.path.join(data_dir, str(idx) + '_y.npy'), y_numpy)
            idx += 1

    data = {'text': X, 'label':Y}
    return data

if __name__ == "__main__":
    
    pass