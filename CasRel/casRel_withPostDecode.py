#!/usr/bin/python
# author kingbone
import torch
from fastNLP import Vocabulary
from transformers import BertTokenizer, AdamW, BertModel
from collections import defaultdict
from random import choice
import json
import torch.nn as nn

import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


class Config:
    """
    句子最长长度是294 这里就不设参数限制长度了,每个batch 自适应长度
    """

    def __init__(self):
        self.bert_path = './PLM/chinese-roberta-wwm-ext/'

        self.num_rel = 3  # 关系的种类数

        self.train_data_path = './er_data/trainWithEntity.json'
        self.dev_data_path = './er_data/devWithEntity.json'
        self.test_data_path = './er_data/devWithEntity.json'

        self.batch_size = 32

        self.rel_dict_path = './ERJE/rel.json'
        id2rel = json.load(open(self.rel_dict_path, encoding='utf8'))
        self.rel_vocab = Vocabulary(unknown=None, padding=None)
        self.rel_vocab.add_word_lst(list(id2rel.values()))  # 关系到id的映射

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.learning_rate = 1e-5
        self.bert_dim = 768
        self.epochs = 1000


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    text = batch[0]
    triple = batch[1]
    entity = batch[2]
    del batch
    return text, triple, entity


class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.dataset = []
        with open(path, encoding='utf8') as F:
            for line in F:
                line = json.loads(line)
                self.dataset.append(line)

    def __getitem__(self, item):
        content = self.dataset[item]
        text = content['text']
        spo_list = content['spo_list']
        entity = content['entity']
        return text, spo_list, entity

    def __len__(self):
        return len(self.dataset)


def create_data_iter(config):
    train_data = MyDataset(config.train_data_path)
    dev_data = MyDataset(config.dev_data_path)
    test_data = MyDataset(config.test_data_path)

    train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_iter = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_iter, dev_iter, test_iter


class Batch:
    def __init__(self, config):
        self.tokenizer = config.tokenizer
        self.num_relations = config.num_rel
        self.rel_vocab = config.rel_vocab
        self.device = config.device

    def __call__(self, text, triple, entities):

        token = self.tokenizer(text, padding=True).data
        batch_size = len(token['input_ids'])
        seq_len = len(token['input_ids'][0])
        sub_head = []
        sub_tail = []
        sub_heads = []
        sub_tails = []
        obj_heads = []
        obj_tails = []
        sub_len = []
        sub_head2tail = []

        for batch_index in range(batch_size):
            inner_input_ids = token['input_ids'][batch_index]  # 单个句子变成索引后
            inner_triples = triple[batch_index]
            entity = entities[batch_index]
            inner_triples = sorted(inner_triples, key=lambda x: len(x['subject']), reverse=True)
            inner_sub_heads, inner_sub_tails, inner_sub_head, inner_sub_tail, inner_sub_head2tail, inner_sub_len, inner_obj_heads, inner_obj_tails = \
                self.create_label(inner_triples, inner_input_ids, seq_len, entity)
            sub_head.append(inner_sub_head)
            sub_tail.append(inner_sub_tail)
            sub_len.append(inner_sub_len)
            sub_head2tail.append(inner_sub_head2tail)
            sub_heads.append(inner_sub_heads)
            sub_tails.append(inner_sub_tails)
            obj_heads.append(inner_obj_heads)
            obj_tails.append(inner_obj_tails)

        input_ids = torch.tensor(token['input_ids']).to(self.device)
        mask = torch.tensor(token['attention_mask']).to(self.device)
        sub_head = torch.stack(sub_head).to(self.device)
        sub_tail = torch.stack(sub_tail).to(self.device)
        sub_heads = torch.stack(sub_heads).to(self.device)
        sub_tails = torch.stack(sub_tails).to(self.device)
        sub_len = torch.stack(sub_len).to(self.device)
        sub_head2tail = torch.stack(sub_head2tail).to(self.device)
        obj_heads = torch.stack(obj_heads).to(self.device)
        obj_tails = torch.stack(obj_tails).to(self.device)

        return {
                   'input_ids': input_ids,
                   'mask': mask,
                   'sub_head2tail': sub_head2tail,
                   'sub_len': sub_len
               }, {
                   'sub_heads': sub_heads,
                   'sub_tails': sub_tails,
                   'obj_heads': obj_heads,
                   'obj_tails': obj_tails
               }

    def create_label(self, inner_triples, inner_input_ids, seq_len, entities):

        inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)
        inner_sub_head, inner_sub_tail = torch.zeros(seq_len), torch.zeros(seq_len)
        inner_obj_heads = torch.zeros((seq_len, self.num_relations))
        inner_obj_tails = torch.zeros((seq_len, self.num_relations))
        inner_sub_head2tail = torch.zeros(seq_len)  # 随机抽取一个实体，从开头一个词到末尾词的索引

        # 因为数据预处理代码还待优化,会有不存在关系三元组的情况，
        # 初始化一个主词的长度为1，即没有主词默认主词长度为1，
        # 防止零除报错,初始化任何非零数字都可以，没有主词分子是全零矩阵
        inner_sub_len = torch.tensor([1], dtype=torch.float)
        # 主词到谓词的映射
        s2ro_map = defaultdict(list)

        for entity in entities:
            entity_id = self.tokenizer(entity['name'], add_special_tokens=False)['input_ids']
            entity_head_idx = self.find_head_idx(inner_input_ids, entity_id)


            if entity_head_idx != -1:
                for entity_idx in entity_head_idx:
                    inner_sub_heads[entity_idx] = 1
                    inner_sub_tails[entity_idx + len(entity_id) - 1] = 1

        for inner_triple in inner_triples:

            inner_triple = (
                self.tokenizer(inner_triple['subject'], add_special_tokens=False)['input_ids'],
                self.rel_vocab.to_index(inner_triple['relation']),
                self.tokenizer(inner_triple['object'], add_special_tokens=False)['input_ids']
            )

            subs_head_idx = self.find_head_idx(inner_input_ids, inner_triple[0])
            objs_head_idx = self.find_head_idx(inner_input_ids, inner_triple[2])
            sub_head_idx = subs_head_idx[0]
            obj_head_idx = objs_head_idx[0]

            if sub_head_idx != -1 and obj_head_idx != -1:
                sub = (sub_head_idx, sub_head_idx + len(inner_triple[0]) - 1)

                for obj_idx in objs_head_idx:
                    s2ro_map[sub].append(
                        (obj_idx, obj_idx + len(inner_triple[2]) - 1, inner_triple[1]))  # {(3,5):[(7,8,0)]} 0是关系

        if s2ro_map:
            sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
            inner_sub_head[sub_head_idx] = 1
            inner_sub_tail[sub_tail_idx] = 1
            inner_sub_head2tail[sub_head_idx:sub_tail_idx + 1] = 1
            inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)
            for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                inner_obj_heads[ro[0]][ro[2]] = 1
                inner_obj_tails[ro[1]][ro[2]] = 1
        
        # inner_sub_heads, inner_sub_tails为实体识别训练的序列标注结果
        return inner_sub_heads, inner_sub_tails, inner_sub_head, inner_sub_tail, inner_sub_head2tail, inner_sub_len, inner_obj_heads, inner_obj_tails

    @staticmethod
    def find_head_idx(source, target):
        target_len = len(target)
        result = []
        for i in range(len(source)):
            if source[i: i + target_len] == target:
                result.append(i)
        if not result:
            result.append(-1)
        return result


class CasRel(nn.Module):
    def __init__(self, config):
        super(CasRel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.sub_heads_linear = nn.Linear(self.config.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.config.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.config.bert_dim, self.config.num_rel)
        self.obj_tails_linear = nn.Linear(self.config.bert_dim, self.config.num_rel)
        self.alpha = 0.25
        self.gamma = 2

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head2tail, sub_len, encoded_text):
        # sub_head_mapping [batch, 1, seq] * encoded_text [batch, seq, dim]
        sub = torch.matmul(sub_head2tail, encoded_text)  # batch size,1,dim
        sub_len = sub_len.unsqueeze(1)
        sub = sub / sub_len  # batch size, 1,dim
        encoded_text = encoded_text + sub
        #  [batch size, seq len,bert_dim] -->[batch size, seq len,relathion counts]
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails

    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        """

        :param token_ids:[batch size, seq len]
        :param mask:[batch size, seq len]
        :param sub_head:[batch size, seq len]
        :param sub_tail:[batch size, seq len]
        :return:
        """
        encoded_text = self.get_encoded_text(input_ids, mask)
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        sub_head2tail = sub_head2tail.unsqueeze(1)  # [[batch size,1, seq len]]
        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head2tail, sub_len, encoded_text)

        return {
            "pred_sub_heads": pred_sub_heads,
            "pred_sub_tails": pred_sub_tails,
            "pred_obj_heads": pred_obj_heads,
            "pred_obj_tails": pre_obj_tails,
            'mask': mask
        }

    def compute_loss(self, pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails, mask, sub_heads,
                     sub_tails, obj_heads, obj_tails):
        rel_count = obj_heads.shape[-1]
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)
        loss_1 = self.loss_fun(pred_sub_heads, sub_heads, mask)
        loss_2 = self.loss_fun(pred_sub_tails, sub_tails, mask)
        loss_3 = self.loss_fun(pred_obj_heads, obj_heads, rel_mask)
        loss_4 = self.loss_fun(pred_obj_tails, obj_tails, rel_mask)
        # return loss_1 + loss_2
        # return loss_3 + loss_4
        return loss_1 + loss_2 + loss_3 + loss_4

    def loss_fun(self, logist, label, mask):
        count = torch.sum(mask)
        logist = logist.view(-1)
        label = label.view(-1)
        mask = mask.view(-1)

        alpha_factor = torch.where(torch.eq(label, 1), 1 - self.alpha, self.alpha)
        focal_weight = torch.where(torch.eq(label, 1), 1 - logist, logist)

        loss = -(torch.log(logist) * label + torch.log(1 - logist) * (1 - label)) * mask
        # return torch.sum(focal_weight * loss) / count
        return torch.sum(loss) / count


def load_model(config):
    device = config.device
    model = CasRel(config)
    model.to(device)

    # prepare optimzier
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=10e-8)
    sheduler = None

    return model, optimizer, sheduler, device


def train_epoch(model, train_iter, dev_iter, optimizer, batch, best_triple_f1, epoch):
    for step, (text, triple, entity) in enumerate(train_iter):
        model.train()
        inputs, labels = batch(text, triple, entity)
        logist = model(**inputs)
        loss = model.compute_loss(**logist, **labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 1:
            sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df = test(model, dev_iter,
                                                                                                     batch)
            print(
                'epoch:{},step:{},sub_precision:{:.4f}, sub_recall:{:.4f}, sub_f1:{:.4f}, triple_precision:{:.4f}, triple_recall:{:.4f}, triple_f1:{:.4f},train loss:{:.4f}'.format(
                    epoch, step, sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1,
                    loss.item()))
            if triple_f1 > best_triple_f1:
                best_triple_f1 = triple_f1
                torch.save(model.state_dict(), 'best_f1_postDecode_newTrainFunc_noGamma_v2.pth')
                print('------------------best model--------------------')
            print(df)

    return best_triple_f1


def train(model, train_iter, dev_iter, optimizer, config):
    epochs = config.epochs
    best_triple_f1 = 0
    for epoch in range(epochs):
        best_triple_f1 = train_epoch(model, train_iter, dev_iter, optimizer, batch, best_triple_f1, epoch)


def test(model, dev_iter, batch):
    model.eval()
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)

    for text, triple, entity in tqdm(dev_iter):
        inputs, labels = batch(text, triple, entity)
        logist = model(**inputs)

        pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])

        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        batch_size = inputs['input_ids'].shape[0]

        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])

        for batch_index in range(batch_size):
            pred_subs = extract_sub(pred_sub_heads[batch_index].squeeze(), pred_sub_tails[batch_index].squeeze())
            true_subs = extract_sub(sub_heads[batch_index].squeeze(), sub_tails[batch_index].squeeze())

            true_objs = extract_objId_and_rel(obj_heads[batch_index], obj_tails[batch_index], inputs['input_ids'][batch_index])
            pred_ojbs = docode_rel(pred_obj_heads[batch_index], pred_obj_tails[batch_index], inputs['input_ids'][batch_index], true_subs)

            pred_ojbs = list(set(pred_ojbs))
            true_objs = list(set(true_objs))


            df['PRED']['sub'] += len(pred_subs)
            df['REAL']['sub'] += len(true_subs)
            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df['TP']['sub'] += 1

            df['PRED']['triple'] += len(pred_ojbs)
            df['REAL']['triple'] += len(true_objs)
            for true_obj in true_objs:
                if true_obj in pred_ojbs:
                    df['TP']['triple'] += 1

    df.loc['sub', 'p'] = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    df.loc['sub', 'r'] = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    df.loc['sub', 'f1'] = 2 * df['p']['sub'] * df['r']['sub'] / (df['p']['sub'] + df['r']['sub'] + 1e-9)

    sub_precision = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    sub_recall = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-9)

    df.loc['triple', 'p'] = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    df.loc['triple', 'r'] = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    df.loc['triple', 'f1'] = 2 * df['p']['triple'] * df['r']['triple'] / (
            df['p']['triple'] + df['r']['triple'] + 1e-9)

    triple_precision = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    triple_recall = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    triple_f1 = 2 * triple_precision * triple_recall / (
            triple_precision + triple_recall + 1e-9)

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


def extract_sub(pred_sub_heads, pred_sub_tails):
    subs = []
    heads = torch.arange(0, len(pred_sub_heads), device=config.device)[pred_sub_heads == 1]
    tails = torch.arange(0, len(pred_sub_tails), device=config.device)[pred_sub_tails == 1]

    for head, tail in zip(heads, tails):
        if tail >= head:
            subs.append((head.item(), tail.item()))
    return subs


def extract_objId_and_rel(obj_heads, obj_tails, input_ids):
    obj_heads = obj_heads.T
    obj_tails = obj_tails.T
    rel_count = obj_heads.shape[0]
    obj_and_rels = []  # [(rel_index,strart_index,end_index),(rel_index,strart_index,end_index)]

    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]

        objs = extract_sub(obj_head, obj_tail)
        if objs:
            for obj in objs:
                start_index, end_index = obj
                ids = get_ids(input_ids, start_index, end_index)
                obj_and_rels.append((rel_index, ''.join(ids)))
    return obj_and_rels


def get_all_possiblity(ids_list):
    re = []
    for i in range(1, len(ids_list) + 1):
        for j in range(len(ids_list) - i + 1):
            re.append(''.join(ids_list[j:j + i]))
    return re


def docode_rel(pred_heads, pred_tails, input_ids, entities_list):
    pred_heads = pred_heads.T
    pred_tails = pred_tails.T
    rel_count = pred_heads.shape[0]
    pred_and_rels = [] # [(rel_index,strart_index,end_index),(rel_index,strart_index,end_index)]

    entities_id_list = [] # [(ids, len(ids))]
    for entity in entities_list:
        e_start, e_end = entity
        ids = get_ids(input_ids, e_start, e_end)
        entities_id_list.append((''.join(ids), len(ids)))
    entities_id_list = sorted(entities_id_list, key=lambda x:x[1])

    for rel_index in range(rel_count):
        pred_head = pred_heads[rel_index]
        pred_tail = pred_tails[rel_index]

        preds = extract_sub(pred_head, pred_tail)
        if preds:
            for pred in preds:
                pos = 0
                start_index, end_index = pred
                ids_list = get_ids(input_ids, start_index, end_index)
                ids = ''.join(ids_list)
                for entity, len_e in entities_id_list:
                    if ids in entity:
                        pos = 1
                        pred_and_rels.append((rel_index, entity))
                        break
                if not pos:
                    pred_and_rels.append((rel_index, ids))  
    return pred_and_rels


def get_ids(input_ids, start, end):
    return [str(id.item()) for id in input_ids[start: end + 1]]


def convert_score_to_zero_one(tensor):
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor


if __name__ == '__main__':
    config = Config()
    train_data = MyDataset(config.train_data_path)

    model, optimizer, sheduler, device = load_model(config)
    train_iter, dev_iter, test_iter = create_data_iter(config)
    batch = Batch(config)
    # for step, (text, triple) in enumerate(train_iter):
    #     inputs, labels = batch(text, triple)
    train(model, train_iter, dev_iter, optimizer, config)
