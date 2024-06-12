from collections import Counter
from FlagEmbedding import FlagModel
from sentence_transformers import util
import json
import torch
import numpy as np
import re
from tqdm import tqdm
import random
import os
from ai.helper import ensure_dir


class BuildFTData():
    def __init__(self):
        self.model = FlagModel('pretrain_models/bge-large-en-v1.5', use_fp16=True)
        self.re_html = re.compile('<[^>]+>')
        self.qids = None
        self.max_length = 256
        self.emb_path = f"vector/bge-large-en-v1.5_embeddings_{self.max_length}"
        self.data_path = "data/bge_ft"
        ensure_dir(self.data_path)
        self.train_file = 'data/qa_train.txt'
        self.bge_train_ft = f'{self.data_path}/bge_ft.json'
        self.corpus = json.load(open('data/pid_to_title_abs_new.json', 'r', encoding='utf8'))
        self.load_embedding()  # 初始化/加载 模型

    def load_sentences(self):
        datas = json.load(open('data/pid_to_title_abs_new.json', 'r', encoding='utf8'))
        qids = []
        contents = []
        for k, v in datas.items():
            title = v['title']
            abstract = v['abstract']
            if title is None:
                title = ''
            if abstract is None:
                abstract = ''
            qids.append(k)
            contents.append(title + " " + abstract)
            # contents.append(title)
        qids = np.array(qids)
        return qids, contents

    def query2context(self, query, topk=20):
        query_embedding = self.model.encode(query, max_length=self.max_length)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(cos_scores, k=topk)
        score = top_results[0].cpu().numpy()[:topk]
        indics = top_results[1].cpu().numpy()
        res = self.qids[indics[: topk]]
        return res, score

    def buildNeg(self, query, pids):
        topk = 600
        ids, scores = self.query2context(query, topk=topk)
        ids = list(ids)
        negs = []
        for id in ids:
            if id not in pids:
                neg_recall = self.corpus.get(id, '')
                if neg_recall != "":
                    title = neg_recall['title']
                    abstract = neg_recall['abstract']
                    if title is None:
                        title = ''
                    if abstract is None:
                        abstract = ''
                    neg = title + " " + abstract
                    negs.append(neg)

        return negs[:20], negs[20:]

    def load_embedding(self):
        qids, contents = self.load_sentences()
        self.qids = qids

        if os.path.exists(self.emb_path):
            self.embeddings = torch.load(self.emb_path)
        else:
            embeddings = self.model.encode(contents,
                                           batch_size=64,
                                           max_length=self.max_length,
                                           )
            self.embeddings = torch.from_numpy(embeddings)
            torch.save(self.embeddings, self.emb_path)

    def create(self):
        fin = open(self.train_file, 'r', encoding='utf8')
        fout = open(self.bge_train_ft, 'w', encoding='utf8')
        total = []
        for idx, i in tqdm(enumerate(fin)):
            i = i.strip()
            data = json.loads(i)
            question = data['question']
            body = data['body']
            pids = data['pids']
            if question is None:
                question = ''
            if body is None:
                body = ''
            body = self.re_html.sub("", body).strip().replace("\n", "")
            query = question + " " + body

            # 召回负样本进行FT
            hard_negs, negs = self.buildNeg(query, pids)

            # 构建正样本
            random.seed(idx)
            for pidx, pid in enumerate(pids):
                relation = self.corpus.get(pid, '')
                if relation != "":
                    title = relation['title']
                    abstract = relation['abstract']
                    if title is None:
                        title = ''
                    if abstract is None:
                        abstract = ''
                    pos = title + " " + abstract
                    random.seed(idx + pidx)
                    # neg_samples1 = random.sample(hard_negs, 2)
                    neg_samples2 = random.sample(negs, 10)
                    line = {"query": query, "pos": [pos], "neg": neg_samples2}
                    total.append(line)

            # break
        random.seed(42)
        random.shuffle(total)
        for line in total:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

        fout.close()


if __name__ == '__main__':
    run = BuildFTData()
    run.create()
