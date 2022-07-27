import re
import string
from collections import defaultdict
import nltk
import numpy as np
import pandas as pd
import razdel
import torch
import umap
import json
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pymystem3 import Mystem
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset  # , Subset
from transformers import AutoModel
from transformers import AutoTokenizer
# --- deploy libs ---
from pymongo import MongoClient
from urllib.parse import quote_plus as quote
from bson.objectid import ObjectId
import warnings
import os
import boto3
import sys

nltk.data.path.append('/app/nltk_data')
warnings.filterwarnings("ignore")

class KeyWords:
    def __init__(self, language='russian'):
        self.punctuation = string.punctuation + '»' + '«'
        self.lang = language
        self.stop_words = stopwords.words(language)

        if language == 'russian':
            self.stem = Mystem()
        else:
            self.stem = SnowballStemmer(language='english')
            self.stop_words += ['i']

    def remove_urls(self, documents) -> list:
        return [re.sub('https?:\/\/.*?[\s+]', '', text) for text in documents]

    def replace_newline(self, documents):
        documents = [text.replace('\n', ' ') + ' ' for text in documents]
        return documents

    def remove_strange_symbols(self, documents):
        return [re.sub(f'[^A-Za-zА-Яа-яё0-9{string.punctuation}\ ]+', ' ', text) for text in documents]

    def tokenize(self, documents) -> list:
        if self.lang == 'english':
            return [nltk.word_tokenize(text) for text in documents]
        else:
            return [[token.text for token in razdel.tokenize(text)] for text in documents]

    def to_lower(self, documents) -> list:
        return [text.lower() for text in documents]

    def remove_punctuation(self, tokenized_documents):
        ttt = set(string.punctuation)
        return [[token for token in tokenized_text if not set(token) < ttt] for tokenized_text in tokenized_documents]

    def remove_numbers(self, documents):
        return [re.sub('(?!:\s)\d\.?\d*', ' ', text) for text in documents]

    def remove_stop_words(self, tokenized_documents) -> list:
        return [[token for token in tokenized_text if token not in self.stop_words] for tokenized_text in
                tokenized_documents]

    def lemmatize(self, documents) -> list:
        if self.lang == 'russian':
            return [''.join(self.stem.lemmatize(text)) for text in documents]
        else:
            return [' '.join(self.stem.stem(token) for token in text.split()) for text in documents]

    def preprocessing(self, documents):
        documents = self.replace_newline(documents)
        documents = self.remove_urls(documents)
        documents = self.remove_strange_symbols(documents)
        documents = self.to_lower(documents)
        documents = self.lemmatize(documents)
        documents = self.remove_numbers(documents)
        tokenized_documents = self.tokenize(documents)
        tokenized_documents = self.remove_stop_words(tokenized_documents)
        tokenized_documents = self.remove_punctuation(tokenized_documents)
        documents = [' '.join(tokenized_text) for tokenized_text in tokenized_documents]
        return documents

    def join_text(self, clean_documents, cluster_ids):
        joined_texts = defaultdict(list)

        for clean_text, cluster_id in zip(clean_documents, cluster_ids):
            joined_texts[cluster_id].append(clean_text)

        for cluster_id, texts in joined_texts.items():
            joined_texts[cluster_id] = ' '.join(texts)
        return list(joined_texts.keys()), list(joined_texts.values())

    def extract_keywords(self, cluster_names, vectorizer, tfidf_matrix, top_n):
        id_2_word = {token_id: word for word, token_id in vectorizer.vocabulary_.items()}
        ind_arr = [list(np.argsort(-x)) for x in tfidf_matrix]

        cluster_keywords = dict()

        for cluster_id, cluster_word_rating, cluster_word_tfidf in zip(cluster_names, ind_arr, tfidf_matrix):
            wl = []
            for word_id in cluster_word_rating[:top_n]:
                wl.append((id_2_word[word_id], cluster_word_tfidf[word_id]))
            cluster_keywords[cluster_id] = wl
        return cluster_keywords

    def get_ctfidf_keywords(self, documents, cluster_ids, top_n=20):
        clean_documents = self.preprocessing(documents)
        cluster_names, joined_texts = self.join_text(clean_documents, cluster_ids)

        vectorizer = CountVectorizer()
        X_counts = vectorizer.fit_transform(joined_texts)

        # calculate c-tf-idf
        df = np.squeeze(np.asarray(X_counts.sum(axis=0)))
        avg_nr_samples = int(X_counts.sum(axis=1).mean())
        idf = np.log((avg_nr_samples / df) + 1)

        tf = X_counts / X_counts.sum(axis=1)
        X_ctfidf = np.multiply(tf, np.expand_dims(idf, axis=0))
        X_ctfidf = np.asarray(X_ctfidf)

        return self.extract_keywords(cluster_names, vectorizer, X_ctfidf, top_n)

    def get_tfifd_v2_keywords(self, documents, cluster_ids, top_n=20):
        clean_documents = self.preprocessing(documents)
        cluster_names, joined_texts = self.join_text(clean_documents, cluster_ids)

        tf_idf_vectorizer = TfidfVectorizer()
        tf_idf_vectorizer.fit(clean_documents)
        idf = tf_idf_vectorizer.idf_
        vocab = tf_idf_vectorizer.vocabulary_

        vectorizer = CountVectorizer()
        X_counts = vectorizer.fit_transform(joined_texts)

        # calculate tf-idf
        tf = X_counts / X_counts.sum(axis=1)
        tf_, idf_ = np.broadcast_arrays(tf, idf)
        X_tfidf = tf_ * idf_

        return self.extract_keywords(cluster_names, vectorizer, X_tfidf, top_n)

class Plot_data():
    def __init__(self):
        pass

    def get_2d_embeddings(self, embeddings, n=2) -> np.array:
        reducer = umap.UMAP(n_components=n)
        umap_data = reducer.fit_transform(embeddings)
        umap_data = StandardScaler().fit_transform(umap_data)
        return umap_data

    def get_cluster_centers(self, umap_data, cluster_ids):
        cluster_names = list(np.unique(cluster_ids))
        centers = np.empty(shape=(len(cluster_names), 3))
        for ind, cluster_name in enumerate(cluster_names):
            x = 0
            y = 0
            size = 0
            for emb, emb_cluster in zip(umap_data, cluster_ids):
                if emb_cluster == cluster_name:
                    x += emb[0]
                    y += emb[1]
                    size += 1
            centers[ind] = [x / size, y / size, size]
        centers[:, 0] += abs(min(centers[:, 0]))
        centers[:, 1] += abs(min(centers[:, 1]))
        return centers

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

class DatasetAutoEnc(Dataset):
    def __init__(self, text, id_=None):
        super().__init__()
        self.id_ = id_
        self.text = text

    def __getitem__(self, idx):
        return self.text['input_ids'][idx], self.text['attention_mask'][idx], -1

    def __len__(self):
        return len(self.text['attention_mask'])

def batch_collate(batch):
    input_ids, attention_mask, label = torch.utils.data._utils.collate.default_collate(batch)
    max_length = attention_mask.sum(dim=1).max().item()
    attention_mask, input_ids = attention_mask[:, :max_length], input_ids[:, :max_length]
    return input_ids, attention_mask, label


def set_grad(net, require=False):
    for param in net.parameters():
        param.requires_grad = require


class AutoEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, n_clusters, intermediate_sizes):
        super().__init__()
        self.embedding_size = embedding_size
        self.alpha = 10
        self.a_enc_centers = nn.Parameter(
            torch.zeros(
                n_clusters,
                embedding_size,
                requires_grad=True,
                dtype=torch.float),
            requires_grad=True
        )
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.intermediate_sizes = intermediate_sizes

        encoder_dim = self.intermediate_sizes + [self.embedding_size]
        decoder_dim = self.intermediate_sizes[::-1] + [self.input_size]
        encoder_activations = [nn.LeakyReLU for _ in range(len(self.intermediate_sizes))] + [None]
        decoder_activations = [nn.LeakyReLU for _ in range(len(self.intermediate_sizes))] + [None]

        self._encoder = nn.Sequential(*self._spec2seq(
            self.input_size, 
            encoder_dim, 
            encoder_activations))
        self._decoder = nn.Sequential(*self._spec2seq(
            self.embedding_size, 
            decoder_dim, 
            decoder_activations))

    def forward(self, x):
        ae_embedding = self._encoder(x)
        reconstructed = self._decoder(ae_embedding)
        return ae_embedding, reconstructed

    def _spec2seq(self, input, dimentions, activations):
        layers = []
        for dim, act in zip(dimentions, activations):
            layer = nn.Linear(input, dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if act:
                layers.append(act())

            input = dim

        return layers


class DeepKMeans(nn.Module):
    def __init__(self, bert, tokenizer, device, embedding_size, n_clusters):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_size = embedding_size
        self.n_clusters = n_clusters
        self.n_epoch = 20
        self.cls_n_epoch = 10
        self.input_size = self.bert.config.hidden_size
        self.ae = AutoEncoder(self.input_size, embedding_size, n_clusters, [500, 500, 2000])
        self.optimizer = optim.AdamW(self.ae.parameters())
        self.criterion = nn.L1Loss().to(self.device)

    def mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):

        output = self.bert(input_ids, attention_mask)
        sentence_embedding = self.mean_pooling(output.last_hidden_state, attention_mask)
        ae_embedding, reconstructed = self.ae(sentence_embedding)
        return ae_embedding, sentence_embedding, reconstructed

    def get_centers(self, loader, val):
        autoEncoderEmbeddings = []
        for i, batch in enumerate(loader):
            input, mask, _ = batch
            autoEncoderEmbedding, bertOutput, x = self.forward(input.to(self.device), mask.to(self.device))
            autoEncoderEmbeddings.append(autoEncoderEmbedding.cpu().detach().numpy())
        autoEncoderEmbeddings = np.concatenate(autoEncoderEmbeddings)
        k_means = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(autoEncoderEmbeddings)
        if val:
            self.validation(loader, kmeans=k_means)
        self.ae.a_enc_centers.copy_(
            torch.as_tensor(
                k_means.cluster_centers_, 
                dtype=torch.float
            ).to(self.device)
        )

    def get_cluster(self, input, mask):

        autoEncoderEmbedding, _, _ = self.forward(input, mask)
        diff = autoEncoderEmbedding.unsqueeze(1) - self.ae.a_enc_centers.unsqueeze(0)
        diff = torch.sum(diff ** 2, dim=-1).unsqueeze(0)
        min_dist = torch.min(diff, dim=-1).indices

        return min_dist, nn.functional.softmax(diff, dim=-1)

    def get_clusters(self, loader):
        y_pred = []
        embs = []
        diffs = []
        for batch in loader:
            input, mask, _ = batch
            cls, diff = self.get_cluster(input.to(self.device), mask.to(self.device))
            diffs.append(torch.squeeze(diff).cpu().detach().numpy())
            y_pred.append(cls.cpu().detach().numpy()[0])
            _, emb, _ = self.forward(input.to(self.device), mask.to(self.device))
            embs.append(emb.cpu().detach().numpy())
        return np.concatenate(y_pred), np.concatenate(embs), np.concatenate(diffs)

    def validation(self, val_loader, kmeans=None, epoch=-1):

        with torch.no_grad():
            self.eval()
            y_pred = []
            y_true = []
            type_ = ''
            for i, batch in enumerate(val_loader):
                input, mask, validation_target = batch
                y_true.append(validation_target.cpu().detach().numpy())
                if kmeans == None:
                    cls, _ = self.get_cluster(input.to(self.device), mask.to(self.device)).cpu().detach().numpy()[0]
                    y_pred.append(cls)
                else:
                    autoEncoderEmbedding, _, _ = self.forward(input.to(self.device), mask.to(self.device))
                    y_pred.append(kmeans.predict(autoEncoderEmbedding.cpu().detach().numpy()))
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            validation_acc = cluster_acc(y_true, y_pred)
            print("Validation ACC", validation_acc)
            validation_ari = adjusted_rand_score(y_true, y_pred)
            print("Validation ARI", validation_ari)
            validation_nmi = normalized_mutual_info_score(y_true, y_pred)
            print("Validation NMI", validation_nmi)
            if len(np.unique(y_pred)) != len(np.unique(y_true)):
                print(f'Есть вырожденный кластер. Всего кластеров {len(np.unique(y_pred))}')
            self.train()

    def cls_train(self, loader, val_lambda, val):
        for epoch in range(self.cls_n_epoch):
            print(f'Epoch: {epoch}')
            for i, batch in enumerate(loader):
                input, mask, _ = batch
                autoEncoderEmbedding, bertOutput, autoEncoderOutout = self.forward(input.to(self.device),
                                                                                   mask.to(self.device))
                diff = autoEncoderEmbedding.unsqueeze(1) - self.ae.a_enc_centers.unsqueeze(0)
                diff = torch.sum(diff ** 2, dim=-1)
                kmeans_loss = torch.mean(torch.max(diff, dim=0).values)
                ae_loss = self.criterion(autoEncoderOutout, bertOutput)
                loss = ae_loss + val_lambda * kmeans_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if val:
                self.validation(loader, epoch=epoch)

    def get_loader_from_excel(self, file_name):
        get_object_response = S3_CLT.get_object(
            Bucket=S3_BUCKET,
            Key=file_name
        )
        dataset = pd.read_excel(get_object_response['Body'].read(), engine='openpyxl').dropna()
        text = self.tokenizer(
            dataset['text'].astype(str).tolist(), 
            max_length=128, 
            truncation=True,
            return_token_type_ids=False,
            padding='max_length', 
            return_tensors='pt')
        return DataLoader(DatasetAutoEnc(text, dataset['id']), batch_size=50, shuffle=False, collate_fn=batch_collate), \
               dataset

    def get_loader_from_text(self, text):
        return DataLoader(DatasetAutoEnc(
            self.tokenizer(text, max_length=128, truncation=True,
                           return_token_type_ids=False,
                           padding='max_length', return_tensors='pt')), batch_size=50, shuffle=False,
            collate_fn=batch_collate)

    def fit(self, loader, val_lambda=0.0001, val=False):
        print('Autoencoder training start')
        for epoch in range(self.n_epoch):
            print(f'Epoch: {epoch}')
            mean_loss = 0
            for i, batch in enumerate(loader):
                input, mask, _ = batch
                autoEncoderEmbedding, bertOutput, autoEncoderOutout = self.forward(input.to(self.device),
                                                                                   mask.to(self.device))
                loss = self.criterion(autoEncoderOutout, bertOutput)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                mean_loss += loss.item()
        print('Getting centers')
        with torch.no_grad():
            self.eval()
            self.get_centers(loader, val=val)
            self.train()
        print('Clusterer trainig start')
        self.cls_train(loader, val_lambda, val=val)


class NewEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.cpu().detach().numpy()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Executer:
    def __init__(self, model_type, n_clusters, file_path):
        bert = AutoModel.from_pretrained(model_type)
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dkm = DeepKMeans(bert, tokenizer, device, n_clusters, n_clusters).to(device)
        self.loader, self.dataset = self.dkm.get_loader_from_excel(file_path)
        self.dkm.fit(self.loader)
        self.dkm.eval()

    def get_payload(self, prj_id):
        try:
            with torch.no_grad():
                cluster_ids, embeddings, _ = self.dkm.get_clusters(self.loader)
            keywords = KeyWords()

            res = keywords.get_ctfidf_keywords(self.dataset['text'].dropna().astype(str), cluster_ids)
            res_v2 = keywords.get_tfifd_v2_keywords(self.dataset['text'].dropna().astype(str), cluster_ids)
            plot = Plot_data()
            umap_data = plot.get_2d_embeddings(embeddings)
            cluster_centers = plot.get_cluster_centers(umap_data, cluster_ids)
            return self.get_json(
                res_v2, 
                self.dataset['text'].dropna().astype(str), 
                umap_data, 
                cluster_ids, 
                self.dataset['id'], 
                1, 
                keywords,
                prj_id
            )
        except Exception as err:
            return {"status": "error", "payload": None}

    def get_json(self, tf_idf, data, umap_data, labels, id_, project_id, keywords, prj_id):

        plot_data_class = Plot_data()
        i_map = plot_data_class.get_cluster_centers(umap_data, labels)
        data = keywords.preprocessing(data)

        answer = {"status": "success", "payload": {"intertopic_map": [], "topics": [], "documents": []}}

        if not data:
            answer = {
                "status": "error",
                "project_id": project_id
            }
            return answer
        for ind, dot in enumerate(i_map):
            answer["payload"]["intertopic_map"].append(
                {
                    "id": labels[ind],
                    "keywords": [x[0] for x in tf_idf[ind]][:5],
                    "size": dot[2],
                    "cord_x": dot[0],
                    "cord_y": dot[1]
                })
        for ind in range(len(labels)):
            answer["payload"]["documents"].append(
                {
                    "_id": prj_id + '_deep_' + str(id_[ind]),
                    "cord_x": umap_data[ind][0],
                    "cord_y": umap_data[ind][1],
                    "cluster_id": labels[ind],
                    "description": [x for x in data[ind].split(" ") if x != " " and len(x) > 3][:7]
                }
            )
        return answer

    def get_topics_by_doc_id(self, prj_id):
        all_responses = []
        try:
            for id_ in self.dataset["id"]:
                document = self.dataset[self.dataset["id"] == id_].dropna().reset_index()["text"].astype(str).tolist()
                loader = self.dkm.get_loader_from_text(document)
                with torch.no_grad():
                    _, _, distribution = self.dkm.get_clusters(self.loader)
                ds =  [{"label": i, "value": value} for i, value in enumerate(distribution[0])]
                top_idx = max(range(len(ds)), key=lambda index: ds[index]['value'])
                top_ds = ds[top_idx]
                response = {
                    "_id": prj_id + '_deep_' + str(id_),
                    "content": document,
                    "distribution": ds,
                    "top_cluster": top_ds
                }
                all_responses.append(response)
            return all_responses
        except Exception as err:
            print('ERROR', err)
            return {"status": "error", "payload": None} 

# ----- S3 credentials -----
S3_BUCKET = os.environ['S3_BUCKET']
SERVICE_NAME = os.environ['SERVICE_NAME']
KEY = os.environ['KEY']
SECRET = os.environ['SECRET']
ENDPOINT = os.environ['ENDPOINT']
SESSION = boto3.session.Session()
S3_CLT = SESSION.client(
    service_name=SERVICE_NAME,
    aws_access_key_id=KEY,
    aws_secret_access_key=SECRET,
    endpoint_url=ENDPOINT
)
S3_RES = SESSION.resource(
    service_name=SERVICE_NAME,
    aws_access_key_id=KEY,
    aws_secret_access_key=SECRET,
    endpoint_url=ENDPOINT
)
# ----- MongoDB credentials -----
MONGODB_HOST = os.environ['MONGODB_HOST']
MONGODB_DATABASE = os.environ['MONGODB_DATABASE']
MONGODB_USERNAME = os.environ['MONGODB_USERNAME']
MONGODB_PASSWORD = os.environ['MONGODB_PASSWORD']

# ---------- MAIN ----------

if __name__ == '__main__':
    input_data = dict(
        prj_id = sys.argv[1],
        file_name_input = sys.argv[2],
        n_clusters = sys.argv[3]
    )
    print(input_data)

    deepkmeans = Executer(
        "DeepPavlov/rubert-base-cased-conversational", 
        int(input_data['n_clusters']),
        input_data['file_name_input']
    )  # тут есть обучение модели и сохранение датасета в поле класса

    payload = deepkmeans.get_payload(prj_id=input_data['prj_id'])['payload']
    print('payload done:', payload.keys())
    result = deepkmeans.get_topics_by_doc_id(prj_id=input_data['prj_id'])
    print('documents done:', result[0])
    payload = json.loads(json.dumps(payload, cls=NewEncoder, ensure_ascii=False))
    result = json.loads(json.dumps(result, cls=NewEncoder, ensure_ascii=False))

    url = 'mongodb://{user}:{pw}@{hosts}/?replicaSet={rs}&authSource={auth_src}'.format(
        user=quote(MONGODB_USERNAME),
        pw=quote(MONGODB_PASSWORD),
        hosts=','.join([MONGODB_HOST]),
        rs='rs01',
        auth_src=MONGODB_DATABASE
    )
    db = MongoClient(url, tlsCAFile='/app/CA.pem')['classify']
    filter = { '_id': ObjectId(input_data['prj_id']) } 
    newValue = { "$set": { 'deep_payload': payload } }
    db['projects'].update_one(filter, newValue)
    db['documents'].insert_many(result)
    print('Deep K-Means finished')
