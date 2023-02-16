#!/usr/bin/python
# author kingbone
import torch
from fastNLP import Vocabulary
from transformers import BertTokenizer, AdamW, BertModel
from collections import defaultdict
from random import choice
import json
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


class Config:
    def __init__(self):
        self.bert_path = './PLM/chinese-roberta-wwm-ext/'

        self.num_rel = 3  # 关系的种类数

        self.train_data_path = './er_data/trainMLC.json'
        self.dev_data_path = './er_data/devMLC_v1.json'
        self.test_data_path = './er_data/devMLC_v1.json'

        self.batch_size = 32

        self.rel_dict_path = './RelationExtraction/MLC/rel.json'
        id2rel = json.load(open(self.rel_dict_path, encoding='utf8'))
        self.rel_vocab = Vocabulary(unknown=None, padding=None)
        self.rel_vocab.add_word_lst(list(id2rel.values()))  # 关系到id的映射

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.learning_rate = 1e-5
        self.bert_dim = 768
        self.epochs = 1000
        self.loss_type = 'bce'


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
        sub_len = []
        sub_head2tail = []
        obj_len = []
        obj_head2tail = []
        label = []

        for batch_index in range(batch_size):
            inner_input_ids = token['input_ids'][batch_index]  # 单个句子变成索引后
            inner_triples = triple[batch_index]
            inner_sub_head2tail, inner_sub_len, inner_obj_head2tail, inner_obj_len, inner_label = self.create_label(inner_triples, inner_input_ids, seq_len)
            sub_len.append(inner_sub_len)
            sub_head2tail.append(inner_sub_head2tail)
            obj_len.append(inner_obj_len)
            obj_head2tail.append(inner_obj_head2tail)
            label.append(inner_label)

        input_ids = torch.tensor(token['input_ids']).to(self.device)
        mask = torch.tensor(token['attention_mask']).to(self.device)
        sub_len = torch.stack(sub_len).to(self.device)
        sub_head2tail = torch.stack(sub_head2tail).to(self.device)
        obj_len = torch.stack(obj_len).to(self.device)
        obj_head2tail = torch.stack(obj_head2tail).to(self.device)
        label = torch.stack(label).to(self.device)

        return {
                   'input_ids': input_ids,
                   'mask': mask,
                   'sub_head2tail': sub_head2tail,
                   'sub_len': sub_len,
                   'obj_head2tail': obj_head2tail,
                   'obj_len': obj_len
               }, {
                   'label': label
               }

    def create_label(self, inner_triples, inner_input_ids, seq_len):

        inner_sub_head2tail = torch.zeros(seq_len)  # 随机抽取一个实体，从开头一个词到末尾词的索引
        inner_obj_head2tail = torch.zeros(seq_len)
        label = torch.zeros(config.num_rel)

        inner_triple_text = choice(inner_triples)
        inner_triple = (
                self.tokenizer(inner_triple_text['subject'], add_special_tokens=False)['input_ids'],
                self.rel_vocab.to_index(inner_triple_text['relation']),
                self.tokenizer(inner_triple_text['object'], add_special_tokens=False)['input_ids']
            )

        subs_head_idx = self.find_head_idx(inner_input_ids, inner_triple[0])
        objs_head_idx = self.find_head_idx(inner_input_ids, inner_triple[2])
        inner_sub_len = torch.tensor([len(inner_triple[0]) * len(subs_head_idx)], dtype=torch.float)
        inner_obj_len = torch.tensor([len(inner_triple[2]) * len(objs_head_idx)], dtype=torch.float)

        for sub_head_idx in subs_head_idx:
            inner_sub_head2tail[sub_head_idx:sub_head_idx + len(inner_triple[0])] = 1
        for obj_head_idx in objs_head_idx:
            inner_obj_head2tail[obj_head_idx:obj_head_idx + len(inner_triple[2])] = 1

        for triple in inner_triples:
            if triple['subject'] == inner_triple_text['subject'] and triple['object'] == inner_triple_text['object']:
                if self.rel_vocab.to_index(inner_triple_text['relation']) != 3:
                    label[self.rel_vocab.to_index(inner_triple_text['relation'])] = 1
        
        return inner_sub_head2tail, inner_sub_len, inner_obj_head2tail, inner_obj_len, label

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


class MLC(nn.Module):
    def __init__(self, config):
        super(MLC, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.linear = nn.Linear(self.config.bert_dim * 2, self.config.num_rel)

    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    def average_pooling(self, sub_head2tail, sub_len, encoded_text):
        # sub_head_mapping [batch, 1, seq] * encoded_text [batch, seq, dim]
        sub = torch.matmul(sub_head2tail, encoded_text)  # batch size,1,dim
        sub_len = sub_len.unsqueeze(1)
        sub = sub / sub_len  # batch size, 1,dim
        return sub

    def forward(self, input_ids, mask, sub_head2tail, sub_len, obj_head2tail, obj_len):
        encoded_text = self.get_encoded_text(input_ids, mask)
        sub_head2tail = sub_head2tail.unsqueeze(1)  # [[batch size,1, seq len]]
        obj_head2tail = obj_head2tail.unsqueeze(1)
        sub_emb = self.average_pooling(sub_head2tail, sub_len, encoded_text) # [batch size, 1, dim]
        obj_emb = self.average_pooling(obj_head2tail, obj_len, encoded_text)
        concat_emb = torch.cat((sub_emb, obj_emb), 2) # [batch size, 1, dim * 2]
        logits = self.linear(concat_emb).squeeze(1) # [batch size, num rel]
        predict = torch.sigmoid(logits)

        return {
            "logits": logits,
            "predict": predict,
        }

    def compute_loss(self, logits, predict, label):
        if self.config.loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(predict, label)
        elif self.config.loss_type == 'softmaxCE':
            loss = self.softmax_ce(predict, label)
        return loss

    def focal_loss(self, logist, label, mask):
        count = torch.sum(mask)
        logist = logist.view(-1)
        label = label.view(-1)
        mask = mask.view(-1)

        alpha_factor = torch.where(torch.eq(label, 1), 1 - self.alpha, self.alpha)
        focal_weight = torch.where(torch.eq(label, 1), 1 - logist, logist)

        loss = -(torch.log(logist) * label + torch.log(1 - logist) * (1 - label)) * mask
        return torch.sum(focal_weight * loss) / count
    
    def softmax_ce(self, logits, label, mask):
        return None


def load_model(config):
    device = config.device
    model = MLC(config)
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
            total_precision, total_recall, total_f1, r0_p, r0_r, r0_f1, r1_p, r1_r, r1_f1, r2_p, r2_r, r2_f1, df = test(model, dev_iter,
                                                                                                     batch)
            print(
                'epoch:{},step:{},total_precision:{:.4f}, total_recall:{:.4f}, total_f1:{:.4f}, train loss:{:.4f}'.format(
                    epoch, step, total_precision, total_recall, total_f1, loss.item()))
            if total_f1 > best_triple_f1:
                best_triple_f1 = total_f1
                torch.save(model.state_dict(), 'MLC.pth')
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
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['triple_total','rel_0','rel_1','rel_2'])
    df.fillna(0, inplace=True)

    for text, triple, entity in tqdm(dev_iter):
        inputs, labels = batch(text, triple, entity)
        logist = model(**inputs)
        batch_size = logist['predict'].shape[0]

        for batch_index in range(batch_size):

            pred = convert_score_to_zero_one(logist['predict'][batch_index])
            label = labels['label'][batch_index]
            df.loc['triple_total']['PRED'] += sum(pred).item()
            df.loc['triple_total']['REAL'] += sum(label).item()

            df.loc['rel_0']['PRED'] += pred[0].item()
            df.loc['rel_0']['REAL'] += label[0].item()
            if pred[0].item() == 1 and label[0].item() == 1:
                df.loc['rel_0']['TP'] += 1
                df.loc['triple_total']['TP'] += 1

            df.loc['rel_1']['PRED'] += pred[1].item()
            df.loc['rel_1']['REAL'] += label[1].item()
            if pred[1].item() == 1 and label[1].item() == 1:
                df.loc['rel_1']['TP'] += 1
                df.loc['triple_total']['TP'] += 1

            df.loc['rel_2']['PRED'] += pred[2].item()
            df.loc['rel_2']['REAL'] += label[2].item()
            if pred[2].item() == 1 and label[2].item() == 1:
                df.loc['rel_2']['TP'] += 1
                df.loc['triple_total']['TP'] += 1

    total_precision, total_recall, total_f1 = calc_prf1('triple_total', df)
    r0_p, r0_r, r0_f1 = calc_prf1('rel_0', df)
    r1_p, r1_r, r1_f1 = calc_prf1('rel_1', df)
    r2_p, r2_r, r2_f1 = calc_prf1('rel_2', df)

    return total_precision, total_recall, total_f1, r0_p, r0_r, r0_f1, r1_p, r1_r, r1_f1, r2_p, r2_r, r2_f1, df


def calc_prf1(name, df):
    df.loc[name, 'p'] = df['TP'][name] / (df['PRED'][name] + 1e-9)
    df.loc[name, 'r'] = df['TP'][name] / (df['REAL'][name] + 1e-9)
    df.loc[name, 'f1'] = 2 * df['p'][name] * df['r'][name] / (df['p'][name] + df['r'][name] + 1e-9)

    total_precision = df['TP'][name] / (df['PRED'][name] + 1e-9)
    total_recall = df['TP'][name] / (df['REAL'][name] + 1e-9)
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-9)
    return total_precision, total_recall, total_f1



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
