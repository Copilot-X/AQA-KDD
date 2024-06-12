from sentence_transformers import SentenceTransformer, util
from FlagEmbedding import FlagModel
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')


class Inference():
    def __init__(self):
        model_path = "output/bge-large-en-v1.5-ft"
        self.model = FlagModel(model_path, use_fp16=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question", }
        self.query_prefix = "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "
        # self.reranker = FlagLLMReranker('/mnt2/pretrained_model/embedding/bge-reranker-v2-gemma', use_fp16=True)
        self.reranker = AutoModel.from_pretrained('pretrain_models/NV-Embed-v1', trust_remote_code=True)
        self.reranker = self.reranker.half()
        self.reranker.to("cuda:0")

        self.re_html = re.compile('<[^>]+>')
        self.qids = None
        self.contents = None
        self.max_length = 256
        self.emb_path = "vector/bge-large-en-v1.5-ft_best"
        self.testFile = "data/AQA-test-public/qa_test_wo_ans_new.txt"
        self.resultFile = "result/nv_result.txt"
        self.corpus = json.load(open('data/AQA-test-public/pid_to_title_abs_update_filter.json', 'r', encoding='utf8'))
        self.load_embedding()  # 初始化/加载 模型

    def load_sentences(self):
        qids = []
        contents = []
        for k, v in self.corpus.items():
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

    def limit_token(self, sentence, max_token=768):
        words = self.tokenizer.tokenize(sentence)
        top_words = words[:max_token]
        # 将单词列表转换为句子
        limit_text = self.tokenizer.convert_tokens_to_string(top_words)
        return limit_text

    def query2context(self, query, topk=20):
        query_embedding = self.model.encode(query, max_length=self.max_length)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(cos_scores, k=topk)
        score = top_results[0].cpu().numpy()[:topk]
        indics = top_results[1].cpu().numpy()
        texts = []
        for idx in indics[: topk]:
            text = self.limit_token(self.contents[idx], max_token=512)
            texts.append(text)
        res = self.qids[indics[: topk]]
        return res, texts, score

    def load_embedding(self):
        qids, contents = self.load_sentences()
        self.qids = qids
        self.contents = contents

        if os.path.exists(self.emb_path):
            self.embeddings = torch.load(self.emb_path)
        else:
            embeddings = self.model.encode(contents,
                                           batch_size=64,
                                           max_length=self.max_length,
                                           )
            self.embeddings = torch.from_numpy(embeddings)
            torch.save(self.embeddings, self.emb_path)

    def predict(self):
        fin = open(self.testFile, 'r', encoding='utf8')
        fout = open(self.resultFile, 'w', encoding='utf8')
        for i in tqdm(fin):
            i = i.strip()
            data = json.loads(i)
            question = data['question']
            body = data['body']
            if question is None:
                question = ''
            if body is None:
                body = ''
            body = self.re_html.sub("", body).strip().replace("\n", " ")
            content = question + " " + body
            ids, texts, _ = self.query2context(content, topk=80)
            # print("=ids=", ids)

            # reranker
            query_text = self.limit_token(content, max_token=256)

            scores = []
            blocks = [texts[i:i + 10] for i in range(0, len(texts), 10)]
            query_embeddings = self.reranker.encode([query_text], instruction=self.query_prefix, max_length=300)
            for block in blocks:
                passage_embeddings = self.reranker.encode(block, instruction="", max_length=512)

                # normalize embeddings
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

                score = (query_embeddings @ passage_embeddings.T)
                score = score.tolist()[0]
                # print(score)
                scores.extend(score)
            id2score = {id: score for id, score in zip(ids, scores)}
            # print(id2score)
            sort_id2score = dict((sorted(id2score.items(), key=lambda x: x[1], reverse=True)))
            sort_ids = list(sort_id2score.keys())

            line = ",".join(sort_ids[:20])
            fout.write(line + "\n")
            # break
        fout.close()


if __name__ == '__main__':
    run = Inference()
    run.predict()