from transformers import BertForMaskedLM, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import defaultdict
import json
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import razdel
from pymystem3 import Mystem
import numpy as np
import pandas as pd
import umap
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

class BertKMeansModel(torch.nn.Module):
    def __init__(self, dataloader_to_init_kmeans, device=torch.device("cpu"), alpha=1, num_clusters=14,
                 model_name="bert-base-uncased", masking_propability=0.1):
        super().__init__()
        masked_lm_bert = BertForMaskedLM.from_pretrained(model_name)
        self.bert = masked_lm_bert.bert
        self.mlm_head = masked_lm_bert.cls
        self.cross_entropy_layer = nn.CrossEntropyLoss(ignore_index=0)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.alpha = alpha
        self.num_clusters = num_clusters
        self.device = device
        self.to(self.device)
        self.init_simple_clustering(dataloader_to_init_kmeans)
        self.masking_propability = masking_propability
        self.vocab_size = masked_lm_bert.config.vocab_size

    def init_simple_clustering(self, dataloader):
        all_embeddings = []
        with torch.no_grad():
            self.bert.eval()
            self.mlm_head.eval()
            for batch in tqdm(dataloader):
                batch = {i: j.to(self.device) for i, j in batch.items()}
                all_embeddings.append(self.forward(**batch)[0].cpu().detach().numpy())
            self.bert.train()
            self.mlm_head.train()

        all_embeddings = np.concatenate(all_embeddings)
        kmeans = KMeans(n_clusters=self.num_clusters, init="k-means++").fit(all_embeddings)
        kmeans.fit(all_embeddings)
        self.centers = nn.Parameter(torch.tensor(kmeans.cluster_centers_))

    def calc_Q(self, Z, centers=None):
        if centers is None:
            centers = self.centers
        residual = Z.unsqueeze(1) - centers.unsqueeze(0)
        dists = (torch.sqrt(torch.sum(residual ** 2, dim=-1)) / self.alpha + 1) ** (-(1 + self.alpha) / 2)
        sum_j = dists.sum(1)
        Q = dists / sum_j.unsqueeze(1)
        return Q

    def calc_P(self, Q):
        f = Q.sum(0)
        P = Q ** 2 / f.unsqueeze(0)
        sum2_j = P.sum(1)
        P = P / sum2_j.unsqueeze(1)
        return P

    def masking(self, input_ids):
        propability = self.masking_propability
        input_ids = input_ids.cpu().detach()
        rand = torch.rand(input_ids.shape)  # .to(device=self.device)
        mask_arr = (rand < propability) * (input_ids != 101) * \
                   (input_ids != 102) * (input_ids != 0)

        labels = input_ids.clone()
        selection = []
        for i in range(input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(input_ids.shape[0]):
            input_ids[i, selection[i]] = 103

        labels *= mask_arr

        return input_ids.to(self.device), labels.to(self.device)

    def mean_pooling(self, output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(output.shape).float()
        sum_embeddings = torch.sum(output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids=None,
                attention_mask=None,
                do_masking=False,
                token_type_ids=None, **kwargs):
        bert_loss = 0

        if do_masking:
            input_ids, labels = self.masking(input_ids)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        if do_masking:
            prediction_scores = self.mlm_head(sequence_output)
            bert_loss = self.cross_entropy_layer(prediction_scores.view(-1, self.vocab_size), labels.view(-1))

        Z = self.mean_pooling(sequence_output, attention_mask)

        return Z, bert_loss


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

        result = dict()
        for i, name in enumerate(cluster_names):
            x, y, size = centers[i, :]
            result[name] = {'x': x, 'y': y, 'size': int(size)}
        return result


class BertKMeansExecutor:
    def __init__(self, tokenizer=None, max_len=128, batch_size=64, lr=3e-5,
                 model_name="DeepPavlov/rubert-base-cased-conversational",
                 num_clusters=8, language='russian'):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.lr = lr
        self.num_clusters = num_clusters
        self.keywords = KeyWords(language)
        self.umapper = Plot_data()


    def train(self, dataloader, epochs=1):
        dataloader_len = len(dataloader)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        total_steps = dataloader_len * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=70,
                                                         num_training_steps=total_steps)
        self.model.bert.train()
        self.model.mlm_head.train()

        for epoch in range(epochs):
            old_centers = self.model.centers.clone().detach()
            tq = dataloader

            for batch in tq:
                self.model.zero_grad()
                batch = {i: j.to(self.device) for i, j in batch.items()}

                Z, bert_loss = self.model(**batch, do_masking=True)
                bert_loss = bert_loss

                real_Q = self.model.calc_Q(Z)
                old_Q = self.model.calc_Q(Z, centers=old_centers)
                old_P = self.model.calc_P(old_Q)
                cluster_loss = self.model.kl_loss(torch.log(real_Q + 1e-08), old_P)

                loss = cluster_loss + bert_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                self.scheduler.step()

    def predict(self, dataloader):
        self.model.bert.eval()
        self.model.mlm_head.eval()
        ids = []
        embedings = []
        y_pred = []
        with torch.no_grad():
            for step, batch in tqdm(enumerate(dataloader)):
                batch = {i: j.to(self.device) for i, j in batch.items()}
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                ids.append(batch["id"])
                output = self.model(**batch)
                Z, _ = output
                real_Q = self.model.calc_Q(Z)
                y_pred.append(torch.argmax(real_Q, dim=1))
                embedings.append(Z)
        self.model.bert.train()
        self.model.mlm_head.train()
        return torch.cat(y_pred).cpu().detach().numpy(), torch.cat(ids).cpu().detach().numpy(), \
               torch.cat(embedings).cpu().detach().numpy()

    def tokenize_data(self, data, ids=None):
        token_function = lambda x: self.tokenizer(x, padding="max_length", truncation=True,
                                                  max_length=self.max_len, return_tensors='pt')
        data = list(map(token_function, data))
        if ids: ids = list(map(int, ids))
        for i in range(len(data)):
            data[i]["input_ids"] = data[i]["input_ids"].squeeze(0)
            data[i]["token_type_ids"] = data[i]["token_type_ids"].squeeze(0)
            data[i]["attention_mask"] = data[i]["attention_mask"].squeeze(0)
            if ids: data[i]["id"] = torch.tensor(ids[i])

        return data

    def load_dataset(self, filename):
        get_object_response = S3_CLT.get_object(
            Bucket=S3_BUCKET,
            Key=filename
        )
        df = pd.read_excel(get_object_response['Body'].read())
        data = df["text"].tolist()
        ids = df["id"].tolist()
        data = self.tokenize_data(data, ids)
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True)
        self.df = df
        return dataloader

    def init_model(self, dataloder):
        self.model = BertKMeansModel(device=self.device, num_clusters=self.num_clusters,
                                     dataloader_to_init_kmeans=dataloder, model_name=self.model_name)
        self.model.to(self.device)

    def get_payload(self, filename, prj_id, do_train=False, epochs=1):
        try:
            response = {"status": "success", "payload": {"intertopic_map": [], "documents": []}}
            dataloader = self.load_dataset(filename)
            self.init_model(dataloader)
            if do_train:
                self.train(dataloader, epochs=epochs)

            y_pred, ids, embedings = self.predict(dataloader)
            coords = self.umapper.get_2d_embeddings(embedings)
            cluster_coords = self.umapper.get_cluster_centers(coords, y_pred)
            key_words = self.keywords.get_ctfidf_keywords(self.df["text"].tolist(), y_pred)
            for cluster, values in cluster_coords.items():
                key_words_of_doc = key_words[cluster]
                if len(key_words_of_doc) > 10: key_words_of_doc = key_words_of_doc[:10]
                response["payload"]["intertopic_map"].append({
                    "id": cluster,
                    "keywords": list(map(lambda x: x[0], key_words_of_doc)),
                    "size": values["size"],
                    "cord_x": values["x"],
                    "cord_y": values["y"]
                })
            self.df["cluster"] = 0
            self.df["x"] = .0
            self.df["y"] = .0

            for i, id_ in enumerate(ids):
                self.df["cluster"][id_] = int(y_pred[i])
                self.df["x"][id_] = int(coords[i][0])
                self.df["y"][id_] = int(coords[i][1])
            for i in range(self.df.shape[0]):
                response["payload"]["documents"].append({"_id": prj_id + '_bert_' + str(int(self.df["id"][i])),
                                                         "cord_x": float(self.df["x"][i]),
                                                         "cord_y": float(self.df["y"][i]),
                                                         "cluster_id": int(self.df["cluster"][i]),
                                                         "description": self.df["text"][i]})
            #return json.dumps(response, cls=NewEncoder, ensure_ascii=False)
            return response
        except Exception as err:
            print('ERROR', err)
            #return json.dumps({"status": "error", "payload": None}, cls=NewEncoder, ensure_ascii=False)

    def get_topics_by_doc_id(self, id_):
        try:
            document = self.df[self.df["id"] == id_]["text"].tolist()
            data = self.tokenize_data(document)[0]
            self.model.bert.eval()
            self.model.mlm_head.eval()
            with torch.no_grad():
                data = {i: j.to(self.device).unsqueeze(0) for i, j in data.items()}
                output = self.model(**data)
                Q = self.model.calc_Q(output[0])
            self.model.bert.train()
            self.model.mlm_head.train()

            distribution = nn.functional.softmax(Q, dim=1)
            ds =  [{"label": i, "value": value} for i, value in enumerate(distribution[0])]
            top_idx = max(range(len(ds)), key=lambda index: ds[index]['value'])
            top_ds = ds[top_idx]
            response = {
                "content": document,
                "distribution": ds,
                "top_cluster": top_ds
            }
            return json.dumps(response, cls=NewEncoder, ensure_ascii=False)
        except Exception as err:
            print('ERROR', err)
            return json.dumps({"status": "error", "payload": None}, cls=NewEncoder, ensure_ascii=False)

    def get_topics_of_documents(self, prj_id):
        response = []
        try:
            for id_ in self.df["id"]:
                distrib = json.loads(self.get_topics_by_doc_id(id_))
                # for every doc -> prj_id + '_lda_' + id = _id
                distrib["_id"] = prj_id + '_bert_' + str(id_)
                #distrib["_id"] = id_
                response.append(distrib)
            return response
        except Exception as err:
            return {"status": "error", "documents": None}

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
def main_dbs():
    input_data = dict(
        prj_id = sys.argv[1],
        file_name_input = sys.argv[2],
    )
    print(input_data)

    b_kmeans = BertKMeansExecutor(
        num_clusters=89,
        max_len=128,
        batch_size=64,
        model_name="DeepPavlov/rubert-base-cased-conversational"
    )
    payload = b_kmeans.get_payload(
        filename=input_data['file_name_input'],
        prj_id=input_data['prj_id'],
        do_train=True,
        epochs=3
    )['payload']
    print('payload done:', payload.keys())
    #docs = b_kmeans.get_topics_by_doc_id(8)
    #print(docs)
    payload = json.loads(json.dumps(payload, cls=NewEncoder))
    result =  b_kmeans.get_topics_of_documents(prj_id=input_data['prj_id'])
    result = json.loads(json.dumps(result, cls=NewEncoder))

    url = 'mongodb://{user}:{pw}@{hosts}/?replicaSet={rs}&authSource={auth_src}'.format(
        user=quote(MONGODB_USERNAME),
        pw=quote(MONGODB_PASSWORD),
        hosts=','.join([MONGODB_HOST]),
        rs='rs01',
        auth_src=MONGODB_DATABASE
    )
    db = MongoClient(url, tlsCAFile='/app/CA.pem')['classify']
    filter = { '_id': ObjectId(input_data['prj_id']) } 
    newValue = { "$set": { 'bert_payload': payload } }
    db['projects'].update_one(filter, newValue)
    db['documents'].insert_many(result)
    print('BERT finished')

if __name__ == "__main__":
    main_dbs()
