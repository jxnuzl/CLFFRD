# scan attention
import copy
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences

from graph_part import GAT3
#import time
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import abc
import torch.nn.utils as utils
from torch_geometric.data import Data
from sklearn.metrics import classification_report, accuracy_score
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
import torch.nn.init as init
import math
import argparse
import pickle
import json
import json, os
import threading
import argparse
import config_file
import random
#from time import *
from PIL import Image
import sys
sys.path.append('/../image_part')
from image_part.resnet import ResNet50
#from image_part.resnet import ResNet50
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import mixgen as mg
import pandas as pd
from vit_pytorch import SimpleViT
import numpy as np
import scipy.stats as stats
from transformers import BertModel
from transformers import BertConfig
import nlpaug.augmenter.word as naw
import json
import http.client
parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="weibo2")
parser.add_argument("-g", "--gpu_id", type=str, default="1")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description")
args = parser.parse_args()

import time
import GAT2
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor
from vit_pytorch.recorder import Recorder

def process_config(config):
    for k,v in config.items():
        config[k] = v[0]
    return config

class PGD(object):

    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.wtrans = nn.Parameter(torch.zeros(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.wtrans.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):

        h = torch.mm(inp, self.W)
        N = h.size()[0]
        Wh1 = torch.mm(h, self.a[:self.out_features, :])
        Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.T)
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        negative_attention = torch.where(adj > 0, -e, zero_vec)
        attention = F.softmax(attention, dim=1)
        negative_attention = -F.softmax(negative_attention,dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        negative_attention = F.dropout(negative_attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, inp)
        h_prime_negative = torch.matmul(negative_attention, inp)
        h_prime_double = torch.cat([h_prime,h_prime_negative],dim=1)
        new_h_prime = torch.mm(h_prime_double,self.wtrans)
        if self.concat:
            return F.elu(new_h_prime)
        else:
            return new_h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


def contrastive_loss(x, x_aug, T):
    """
    :param x: the hidden vectors of original data
    :param x_aug: the positive vector of the auged data
    :param T: temperature
    :return: loss
    """
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss





class Signed_GAT(nn.Module):
    def __init__(self,node_embedding,cosmatrix,nfeat, uV, original_adj, hidden = 16, \
                                            nb_heads = 4, n_output = 300, dropout = 0, alpha = 0.3):

        super(Signed_GAT, self).__init__()
        self.dropout = dropout
        self.uV = uV
        embedding_dim = 300
        self.user_tweet_embedding = nn.Embedding(num_embeddings=self.uV, embedding_dim=embedding_dim, padding_idx=0)
        self.user_tweet_embedding.from_pretrained(torch.from_numpy(node_embedding))
        self.original_adj = torch.from_numpy(original_adj.astype(np.float64)).cuda()
        self.potentinal_adj = torch.where(cosmatrix>0.5,torch.ones_like(cosmatrix),torch.zeros_like(cosmatrix)).cuda()
        self.adj = self.original_adj + self.potentinal_adj
        self.adj = torch.where(self.adj>0,torch.ones_like(self.adj),torch.zeros_like(self.adj))

        self.attentions = [GraphAttentionLayer(nfeat, n_output, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nb_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nfeat * nb_heads, n_output, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, X_tid):
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda()).to(torch.float32)
        x = F.dropout(X, self.dropout, training=self.training)
        adj = self.adj.to(torch.float32)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(self.out_att(x, adj))
        return x[X_tid]

class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)
        return V_att


    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output


    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output
class Rumor_data(Dataset):
    def __init__(self,X_train_tid, X_train, y_train,
           train_content,train_object,train_enity):
        self.id=X_train_tid
        self.train=X_train
        self.y=y_train
        self.content=train_content
        self.object=train_object
        self.enity=train_enity
    def __len__(self):
        return (len(self.y))
    def __getitem__(self, item):
        return self.id[item],self.train[item],self.y[item],self.content[item],self.object[item],self.enity[item]
def format_time(time):
    elapsed_rounded=int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.newid2imgnum = config['newid2imgnum']
        self.trans = self.img_trans()
        self.path = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_images/weibo_images_all/'

    def img_trans(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform

    @abc.abstractmethod
    def forward(self):
        pass

    def mfan(self, x_tid, x_text, y, loss, i, total, params,pgd_word,content,object,image,enity):
        self.optimizer.zero_grad()
        logit_defense,dist3,dist4= self.forward(x_tid, x_text,content,object,image,enity)
        loss_classification = loss(logit_defense, y)
        loss_mse = nn.MSELoss()


        #loss_dis3=loss_mse(dist3[0],dist3[1])

        loss_defense = loss_classification+dist3+dist4
        loss_defense.backward()

        K = 3
        pgd_word.backup_grad()
        for t in range(K):
            pgd_word.attack(is_first_attack=(t == 0))
            if t != K - 1:
                self.zero_grad()
            else:
                pgd_word.restore_grad()
            loss_adv,dist3,dist4= self.forward(x_tid, x_text,content,object,image,enity)
            loss_adv = loss(loss_adv,y)
            loss_adv.backward()
        pgd_word.restore()

        self.optimizer.step()
        corrects = (torch.max(logit_defense, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print(
            'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                                                       loss_defense.item(),
                                                                                                       accuracy,
                                                                                                       corrects,
                                                                                                       y.size(0)))

    def fit(self, X_train_tid, X_train, y_train,
            X_dev_tid, X_dev, y_dev,train_content,dev_content,train_object,dev_object,train_enity,dev_enity):

        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3, weight_decay=0)

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_train_tid=X_train_tid.cuda()
        X_train=X_train.cuda()
        y_train=y_train.cuda()
        train_object=train_object.cuda()
        #train_contents=train_content
        #dataset = TensorDataset(X_train_tid, X_train, y_train)
        #dataset = TensorDataset(X_train_tid, X_train, y_train)
        #dataset=Dataset(X_train_tid,X_train,y_train,train_contents)
        dataset=Rumor_data(X_train_tid, X_train, y_train,
           train_content,train_object,train_enity)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loss = nn.CrossEntropyLoss()
        params = [(name, param) for name, param in self.named_parameters()]
        params = [(name, param) for name, param in self.named_parameters()]
        params = [(name, param) for name, param in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        trainable_pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'{trainable_pytorch_total_params:,}trainable_pytorch_total_params.')
        print(f'{trainable_pytorch_total_params / (1024 * 1024):.2f}M trainable_pytorch_total_params')
        # total_params += sum(p.numel() for p in params.buffers())
        print(f'{total_params:,}total_params.')
        print(f'{total_params / (1024 * 1024):.2f}M total parameters')
        pgd_word = PGD(self, emb_name='word_embedding', epsilon=6, alpha=1.8)
        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch + 1, "/", self.config['epochs'])
            self.train()
            avg_loss = 0
            avg_acc = 0
            to = time.time()
            for i, data in enumerate(dataloader):
                img_list=[]

                total = len(dataloader)
                #batch_x_tid, batch_x_text, batch_y,batch_content = (item.cuda(device=self.device) for item in data)
                batch_x_tid, batch_x_text, batch_y,batch_content,batch_object,batch_enity = (item for item in data)

                for newid in batch_x_tid.cpu().numpy():
                    imgnum = self.newid2imgnum[newid]
                    imgpath = self.path + imgnum + '.jpg'
                    im = np.array(self.trans(Image.open(imgpath)))
                    im = torch.from_numpy(np.expand_dims(im, axis=0)).to(torch.float32)
                    img_list.append(im)


                self.mfan(batch_x_tid, batch_x_text, batch_y, loss, i, total, params,pgd_word,batch_content,batch_object,img_list,batch_enity)




                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
            t1 = time.time()
            traing_time = t1 - to
            traing_time = format_time(traing_time)
            print(traing_time)
            self.evaluate(X_dev_tid, X_dev, y_dev,dev_content,dev_object,dev_enity)

    def evaluate(self, X_dev_tid, X_dev, y_dev,dev_content,dev_object,dev_enity):
        y_pred = self.predict(X_dev_tid, X_dev,y_dev,dev_content,dev_object,dev_enity)
        acc = accuracy_score(y_dev, y_pred)

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("save model!!!   at ",self.config['save_path'])

    def predict(self, X_test_tid, X_test,y_test,test_content,test_object,test_enity):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid).cuda()
        X_test = torch.LongTensor(X_test).cuda()
        #test_object=torch.LongTensor(test_object).cuda()
        test_object=test_object.cuda()


        #dataset = TensorDataset(X_test_tid, X_test)
        dataset=Rumor_data(X_test_tid,X_test,y_test,test_content,test_object,test_enity)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            img_list=[]
            with torch.no_grad():
                #batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)
                batch_x_tid, batch_x_text,batch_y,batch_content,batch_object,batch_enity = (item for item in data)
                for newid in batch_x_tid.cpu().numpy():
                    imgnum = self.newid2imgnum[newid]
                    imgpath = self.path + imgnum + '.jpg'
                    im = np.array(self.trans(Image.open(imgpath)))
                    im = torch.from_numpy(np.expand_dims(im, axis=0)).to(torch.float32)
                    img_list.append(im)
                logits,dist3,dist4= self.forward(batch_x_tid, batch_x_text,batch_content,batch_object,img_list,batch_enity)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred
def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
#scan attention
def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn*smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext

class simclr(nn.Module):
    def __init__(self, hidden_dim, dataset_num_features, dataset_num, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.dataset_num = dataset_num
        self.embedding_dim = 300
        #self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    @torch.no_grad()
    def sample_negative_index(self, negative_number, epoch, epochs):

        lamda = 1/2
        # lamda = 1
        #lamda = 2
        lower, upper = 0, self.dataset_num
        mu_1 = ((epoch-1) / epochs) ** lamda * (upper - lower)
        mu_2 = ((epoch) / epochs) ** lamda * (upper - lower)
        # sigma = negative_number / 6
        # # X表示含有最大最小值约束的正态分布
        # X = stats.truncnorm(
        #     (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数 正态分布采样
        # X = stats.uniform(mu_1,mu_2-mu_1)  # 均匀分布采样
        X = stats.uniform(1,mu_2)
        index = X.rvs(negative_number)  # 采样
        index = index.astype(np.int)
        return index

    #def forward(self, x, edge_index, batch):

        # batch_size = data.num_graphs
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #if x is None:
            #x = torch.ones(batch.shape[0]).to(device)

        #y, M = self.encoder(x, edge_index, batch)

        #y = self.proj_head(y)

        #return y

    def rank_negative_queue(self, x1, x2):

        x2 = x2.t()
        x = x1.mm(x2)

        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)

        final_value = x.mul(1 / x_frobenins)

        sort_queue, _ = torch.sort(final_value, dim=0, descending=False)

        return sort_queue

    def loss_cal(self, q_batch, q_aug_batch, negative_sim):

        T = 0.2

        # q_batch = q_batch[: q_aug_batch.size()[0]]

        positive_sim = torch.cosine_similarity(q_batch, q_aug_batch, dim=1)  # 维度有时对不齐

        positive_exp = torch.exp(positive_sim / T)

        negative_exp = torch.exp(negative_sim / T)

        negative_sum = torch.sum(negative_exp, dim=0)

        loss = positive_exp / (positive_exp+negative_sum)

        loss = -torch.log(loss).mean()

        return loss







class resnet50():
    def __init__(self):
        self.newid2imgnum = config['newid2imgnum']
        self.model = models.resnet50(pretrained=True).cuda()
        self.model.fc = nn.Linear(2048, 300).cuda()
        torch.nn.init.eye_(self.model.fc.weight)
        self.path = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_images/weibo_images_all/'
        self.trans = self.img_trans()
    def img_trans(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
        return transform
    def forward(self,xtid):
        img_path = []
        img_list = []
        image_name=[]
        for newid in xtid.cpu().numpy():
            imgnum = self.newid2imgnum[newid]
            imgpath = self.path + imgnum + '.jpg'
            im = np.array(self.trans(Image.open(imgpath)))
            im = torch.from_numpy(np.expand_dims(im, axis=0)).to(torch.float32)
            img_list.append(im)
            image_name.append(imgpath)
        batch_img = torch.cat(img_list, dim=0).cuda()
        img_output = self.model(batch_img)
        return img_output,image_name
tokenizer = BertTokenizer.from_pretrained('/home/newDisk/zl/pytorch code/text_image_gcn2/data/bert-base-uncased', do_lower_case=True)
def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
def get_token_ids(x_train):
    token_tr = []
    token_tst = []
    count = 0
    for sent in x_train:
        tokens = tokenizer.encode(sent, add_special_tokens=True, max_length=512)
        token_tr.append(tokens)
        count += 1
        if (count % 1000 == 0):
            print(count)

    #for sent1 in x_test:
        #tokens1 = tokenizer.encode(sent1, add_special_tokens=True, max_length=512)
        #token_tst.append(tokens1)
        #count += 1
        #if (count % 1000 == 0):
            #print(count)

    #return token_tr, token_tst
    return token_tr

class BertMapping(nn.Module):
    """
    """
    def __init__(self):
        super(BertMapping, self).__init__()
        bert_config = BertConfig.from_pretrained('/home/newDisk/zl/pytorch code/text_image_gcn2/data/bert-base-uncased')
        self.bert = BertModel(bert_config)
        freeze_layers(self.bert)
        final_dims = 256
        Ks = [1, 2, 3]
        in_channel = 1
        out_channel = 512
        embedding_dim = bert_config.hidden_size
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embedding_dim)) for K in Ks])
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.mapping = nn.Linear(len(Ks)*out_channel, final_dims)
        self.cls_layer = nn.Linear(final_dims, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids,attention_mask=attention_mask, return_dict=True)
        x = outputs.last_hidden_state
        #code = torch.mean(x, dim=1)
        #x = outputs.last_hidden_state.unsqueeze(1)  # (batch_size, 1, token_num, embedding_dim)
        #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(batch_size, out_channel, W), ...]*len(Ks)
        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        #output = torch.cat(x, 1)
        #output = self.dropout(output)
        #code = self.mapping(output)
        # code = F.tanh(code)
        #feats = F.normalize(code, p=2, dim=1)
        #code = self.cls_layer(feats)
        #code = F.softmax(code, dim=1)
        #return code
        return x

def word2vec(post, word_id_map, W,sequence_len):
    word_embedding = []
    mask = []
    # length = []

    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) - 1
        mask_seq = np.zeros(sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < sequence_len:
            sen_embedding.append(0)

        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        # length.append(seq_len)
    return word_embedding, mask

object_vector_path = '/home/newDisk/zl/pytorch code/MFAN/dataset/weibo/weibo_files/enity_ebd2.pickle'
f = open(object_vector_path, 'rb')
weight = pickle.load(f)
W, W2, word_idx_map, vocab, max_len = weight[0], weight[1], weight[2], weight[3], weight[4]
vocab_size = len(vocab)
sequence_len = max_len
class GATopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2

class MFAN(NeuralNetwork):
    def __init__(self, config, adj, original_adj):
        super(MFAN, self).__init__()
        self.config = config
        self.uV = adj.shape[0]
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        self.mh_attention = TransformerBlock(input_size=300, n_heads=8, attn_dropout=0)
        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))
        self.cosmatrix = self.calculate_cos_matrix()
        self.gat_relation = Signed_GAT(node_embedding=config['node_embedding'], cosmatrix = self.cosmatrix,nfeat=300, \
                                      uV=self.uV, nb_heads=1,
                                      original_adj=original_adj, dropout=0)

        #self.cosmatrix2=self.calculate_cos_matrix2(config['node_embedding2'])

        #self.gat_relation2=Signed_GAT(node_embedding=config['node_embedding2'],cosmatrix=self.cosmatrix2,nfeat=768,\
                                      #uV=)


        self.image_embedding = resnet50()
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        #self.fc3 = nn.Linear(1800,900)
        self.fc3=nn.Linear(512,600)
        self.fc4 = nn.Linear(600,600)
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=config['num_classes'])
        #self.fc2 = nn.Linear(in_features=600, out_features=config['num_classes'])
        self.alignfc_g = nn.Linear(in_features=300, out_features=300)
        self.alignfc_t = nn.Linear(in_features=300, out_features=300)
        self.alignfc_i=nn.Linear(in_features=300,out_features=300)

        self.init_weight()
        self.imagefc=nn.Linear(512,300)
        self.textfc=nn.Linear(512,300)
        self.image_text=nn.Linear(600,300)
        self.objectfc=nn.Linear(2048,256)

        #self.lstm=nn.LSTM(32, 150, batch_first=True, bidirectional=True)
        self.berts=BertMapping()
        self.bertfc=nn.Linear(768,300)
        self.newfc=nn.Linear(600,256)


        self.imagefc2=nn.Linear(1024,300)
        self.texfc=nn.Linear(768,256)
        self.model = models.resnet50(pretrained=True).cuda()
        torch.nn.init.eye_(self.model.fc.weight)
        self.model.fc = nn.Linear(2048, 300).cuda()
        self.embed = nn.Embedding(vocab_size, 32)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))

        config_img = GATopt(256, 1)
        config_cap = GATopt(256, 1)
        config_rcnn = GATopt(256, 1)
        config_joint = GATopt(256, 1)
        self.gat_1 = GAT3.GAT(config_rcnn)
        self.gat_2 = GAT3.GAT(config_img)
        # self.gat_cap = GAT3.GAT(config_cap)
        self.gat_cap = GAT3.GAT(config_cap)
        self.gat_cap2 = GAT3.GAT(config_cap)
        self.gat_cat_2 = GAT3.GAT(config_joint)

        # self.gat_cat_1 = GAT3.GAT(config_joint)
        self.gat_cat_1 = GAT3.GAT(config_joint)
        #self.gat_cat_2 = GAT3.GAT(config_joint)
        self.v = ViT(
            image_size=224,
            patch_size=32,
            num_classes=256,
            dim=256,
            depth=6,
            heads=16,
            mlp_dim=256
        )
        self.v2 = Recorder(self.v)
        self.v3 = Extractor(self.v)
        self.lstm = nn.LSTM(300, 128, batch_first=True, bidirectional=True)

    def calculate_cos_matrix(self):
        a,b = torch.from_numpy(config['node_embedding']),torch.from_numpy(config['node_embedding'].T)
        c = torch.mm(a, b)
        aa = torch.mul(a, a)
        bb = torch.mul(b, b)
        asum = torch.sqrt(torch.sum(aa, dim=1, keepdim=True))
        bsum = torch.sqrt(torch.sum(bb, dim=0, keepdim=True))
        norm = torch.mm(asum, bsum)
        res = torch.div(c, norm)
        return res

    def calculate_cos_matrix2(self,node):
        a,b = torch.from_numpy(node),torch.from_numpy(node.T)
        c = torch.mm(a, b)
        aa = torch.mul(a, a)
        bb = torch.mul(b, b)
        asum = torch.sqrt(torch.sum(aa, dim=1, keepdim=True))
        bsum = torch.sqrt(torch.sum(bb, dim=0, keepdim=True))
        norm = torch.mm(asum, bsum)
        res = torch.div(c, norm)
        return res

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)

    def forward(self, X_tid, X_text,content,object,new_image,enity):
        X_text = self.word_embedding(X_text)
        blip_imgs=[]
        blip_tex=[]
        if self.config['user_self_attention'] == True:
            X_text = self.mh_attention(X_text, X_text, X_text)
        #g_text=X_text
        #out,hidden=self.lstm(g_text)
        #g_texts=out
        new_text=X_text
        X_text = X_text.permute(0, 2, 1)
        rembedding = self.gat_relation.forward(X_tid)




        iembedding,image_names = self.image_embedding.forward(X_tid)
        image = torch.cat(new_image, dim=0).cuda()
        logits, image = self.v3(image)
        #image=self.imagefc2(image)

        out, hidden = self.lstm(new_text)
        cap_emb = out

        #cap_gat = self.gat_cap2(cap_emb)
        # cap_gat=self.gat_cap2(cap_emb)
        #cap_embs = l2norm(torch.mean(cap_gat, dim=1))
        #cap_embs = F.leaky_relu(cap_embs)



        # image feature image_names zuowei tupiandemingzi

        #image = l2norm(image, dim=-1)
        #text = l2norm(text, dim=-1)

        #g_image_text = torch.cat((text, image), dim=1)
        #g_image_text=self.image_text(g_image_text)

        # faster rcnn (batch_size,36,300)
        image_object=self.objectfc(object)
        #rcnn_emb = self.gat_1(image_object)
        #image_object=torch.mean(image_object,dim=1)
        #image_object=l2norm(image_object,dim=-1)


        # bert tiqu shangxiawen tezheng


        #image2=torch.cat(new_image,dim=0).cuda()
        #batch_content = list(content)
        #new_image, new_text = mg.mixgen_batch(image2, batch_content, num=16)
        # aug = naw.ContextualWordEmbsAug(
        # model_path='bert-base-chinese', action='substitute'
        # )
        # new_texts = []
        # for i in range(len(batch_content)):
        # augmented_text = aug.augment(batch_content[i])
        # new_texts.append(augmented_text)

        # imgs = []
        # suiji caijian zengqiangtupian
        # transform = transforms.RandomCrop(200)
        # transform = transforms.RandomHorizontalFlip(p=0.5)
        # for i in range(len(new_image)):
        # img = transform(new_image[i])
        # imgs.append(img)
        # new_images = torch.cat(imgs, dim=0).cuda()

        #new_image = self.v(new_image)
        #new_image = self.imagefc2(new_image)
        #new_image=self.model(new_image)
        #new_image=self.model.fc(new_image)
        #new_image = l2norm(new_image, dim=-1)
        #text_list = new_text

        conv_block = [rembedding]
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = torch.squeeze(pool)
            conv_block.append(pool)

        conv_feature = torch.cat(conv_block, dim=1)
        graph_feature,text_feature = conv_feature[:,:300],conv_feature[:,300:]

        glo_feature=torch.cat((text_feature,iembedding),1)
        glo_feature=self.newfc(glo_feature)
        # tongguo tupian dui wenbenjiaquan

        #text_enity=enity

        #word_embedding, mask = word2vec(text_enity, word_idx_map, W,sequence_len)
        #word_embedding=torch.from_numpy(np.array(word_embedding)).cuda()
        #texts = self.embed(word_embedding)
        #mask=torch.from_numpy(np.array(mask)).cuda()
        #texts = texts * mask.unsqueeze(2).expand_as(texts)
        #out, hidden = self.lstm(texts)
        #text_enity=out
        #text_enity=torch.mean(text_enity,dim=1)
        #text_enity = self.mh_attention(text_enity,text_enity, text_enity)
        #text_enity = torch.mean(text_enity, dim=1)
        #text_enity = l2norm(text_enity, dim=-1)
        #text_enity = enity
        text_enity=enity
        xtr_token2 = get_token_ids(text_enity)
        xtr_token2 = pad_sequences(xtr_token2, maxlen=200, dtype="long",
                                   value=0, truncating="post", padding="post")
        attention_mask_tr2 = []
        for sent in xtr_token2:
            att_mask2 = [int(token_id > 0) for token_id in sent]
            attention_mask_tr2.append(att_mask2)
        train_input_text2 = torch.tensor(xtr_token2)
        train_input_text2 = train_input_text2.cuda()
        train_input_mask2 = torch.tensor(attention_mask_tr2)
        train_input_mask2 = train_input_mask2.cuda()
        text_bert2 = self.berts(train_input_text2, train_input_mask2)
        text_enity = text_bert2
        # g_texts2=torch.mean(g_texts2,dim=1)
        # g_texts2=l2norm(g_texts2,dim=-1)
        text_enity = self.texfc(text_enity)
        # g_texts2 = l2norm(g_texts2, dim=-1)
        # tongguo transformer
        # text_enity = torch.mean(text_enity, dim=1)
        context_img = SCAN_attention(text_enity, image_object, smooth=0.9)
        # context_img = torch.pow(torch.sub(context_img, g_texts), 2)
        context_img = torch.mul(context_img, text_enity)
        context_img = l2norm(torch.mean(context_img, dim=1), dim=-1)

        img_context = SCAN_attention(image_object, text_enity, smooth=0.9)
        # img_context=torch.pow(torch.sub(img_context,image_object),2)
        img_context = torch.mul(img_context, image_object)
        img_context = l2norm(torch.mean(img_context, dim=1), dim=-1)



        #enity_object = torch.cat((image_object, text_enity), 1)
        # enity_object = SCAN_attention(image_object,text_enity,smooth=0.9)
        #enity_object = self.gat_cat_2(enity_object)
        #image=self.gat_2(image)


        img_txt_cat4 = torch.cat((cap_emb, image), 1)
        #img_txt_cat4 = torch.cat((img_txt_cat4, enity_object), 1)
        img_txt_cat4 = self.gat_cat_1(img_txt_cat4)
        image_texts = F.leaky_relu(torch.mean(img_txt_cat4, dim=1))
        image_text=torch.cat((image_texts,glo_feature),1)

        dist3 = contrastive_loss(context_img, img_context, 0.5)
        dist4=contrastive_loss(text_feature,iembedding,0.5)




        #context_img=SCAN_attention(text_enity,image_object,smooth=0.9)
        #context_img = torch.pow(torch.sub(context_img, text_enity), 2)
        #context_img=l2norm(torch.mean(context_img,dim=1),dim=-1)
        #tongguo wenben dui tupian jiaquan

        #img_context=SCAN_attention(image_object,text_enity,smooth=0.9)
        #img_context=torch.pow(torch.sub(img_context,image_object),2)
        #img_context=l2norm(torch.mean(img_context,dim=1),dim=-1)


        #text_feature=text
        #iembedding=image

        #image feature



        # co-attention enity object
        #self_att_ts = self.mh_attention(text_enity.view(bsz, -1, 300), text_enity.view(bsz, -1, 300), \
                                       #text_enity.view(bsz, -1, 300))
        #self_att_is = self.mh_attention(image_object.view(bsz, -1, 300), image_object.view(bsz, -1, 300), \
                                       #image_object.view(bsz, -1, 300))
        #co_att_ti = self.mh_attention(self_att_ts, self_att_is, self_att_is).view(bsz, 300)
        #co_att_it = self.mh_attention(self_att_is, self_att_ts, self_att_ts).view(bsz, 300)




        #text_enhanced = self.mh_attention(self_att_i.view((bsz,-1,300)), self_att_t.view((bsz,-1,300)),\
                                          #self_att_t.view((bsz,-1,300))).view(bsz, 300)

        #align_text = self.alignfc_t(text_enhanced)




        #self_att_t = text_enhanced.view((bsz, -1, 300))
        #co_att_tg = self.mh_attention(self_att_t, self_att_g, self_att_g).view(bsz, 300)
        #co_att_gt = self.mh_attention(self_att_g, self_att_t, self_att_t).view(bsz, 300)
        #co_att_ti = self.mh_attention(self_att_t, self_att_i, self_att_i).view(bsz, 300)
        #co_att_it = self.mh_attention(self_att_i, self_att_t, self_att_t).view(bsz, 300)
        #co_att_gi = self.mh_attention(self_att_g, self_att_i, self_att_i).view(bsz, 300)
        #co_att_ig = self.mh_attention(self_att_i, self_att_g, self_att_g).view(bsz, 300)


        #att_feature = torch.cat((co_att_tg, co_att_gt, co_att_ti, co_att_it, co_att_gi, co_att_ig), dim=1)
        #att_feature=torch.cat((self_att_t,self_att_g,self_att_i,context_img,img_context),dim=1)

        #a1 = self.relu(self.dropout(self.fc3(att_feature)))
        #a1 = self.relu(self.fc4(a1))
        #a1 = self.relu(self.fc1(a1))
        #d1 = self.dropout(a1)
        #d1=torch.cat((d1,g_image_text),dim=1)
        #output = self.relu(self.dropout(self.fc2(image_text)))
        att_feature = image_text
        a1 = self.relu(self.dropout(self.fc3(att_feature)))
        a1 = self.relu(self.fc4(a1))
        a1 = self.relu(self.fc1(a1))
        d1 = self.dropout(a1)
        output = self.fc2(d1)

        return output,dist3,dist4
        #return output, dist, dist2

def load_dataset():
    pre = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_files'
    train_content=[]
    dev_content=[]
    test_content=[]
    #train_content=
    #df = pd.read_csv('/home/newDisk/zl/gitcode/MFAN/dataset/weibo/weibo_files/weibo.train', sep='\t')
    # print(df)
    #for i in range(len(df)):
        #train_content.append(df.iloc[i, 1])
    #df2=pd.read_csv('/home/newDisk/zl/gitcode/MFAN/dataset/weibo/weibo_files/weibo.dev', sep='\t')
    #for i in range(len(df2)):
        #dev_content.append(df2.iloc[i,1])

    #df3=pd.read_csv('/home/newDisk/zl/gitcode/MFAN/dataset/weibo/weibo_files/weibo.test', sep='\t')
    #for i in range(len(df3)):
        #test_content.append(df3.iloc[i,1])
    with open( "/home/newDisk/zl/gitcode/MFAN/dataset/weibo/weibo_files/weibo.train", 'r', encoding='utf-8') as input:
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            train_content.append(content)
    with open("/home/newDisk/zl/gitcode/MFAN/dataset/weibo/weibo_files/weibo.dev", 'r', encoding='utf-8') as input:
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            dev_content.append(content)
    with open("/home/newDisk/zl/gitcode/MFAN/dataset/weibo/weibo_files/weibo.test", 'r', encoding='utf-8') as input:
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            test_content.append(content)

    train_object=np.load('/home/newDisk/zl/gitcode/Faster-R-CNN-with-model-pretrained-on-Visual-Genome/train_object.npy',encoding='latin1')
    dev_object=np.load('/home/newDisk/zl/gitcode/Faster-R-CNN-with-model-pretrained-on-Visual-Genome/dev_object.npy',encoding='latin1')
    test_object=np.load('/home/newDisk/zl/gitcode/Faster-R-CNN-with-model-pretrained-on-Visual-Genome/test_object.npy',encoding='latin1')
    train_vocab = open('/home/newDisk/zl/pytorch code/MFAN/dataset/weibo/weibo_files/new_train_enity.pickle', 'rb')
    train_enitys = pickle.load(train_vocab, encoding='latin1')
    dev_vocab = open('/home/newDisk/zl/pytorch code/MFAN/dataset/weibo/weibo_files/new_dev_enity.pickle', 'rb')
    dev_enitys = pickle.load(dev_vocab, encoding='latin1')
    test_vocab = open('/home/newDisk/zl/pytorch code/MFAN/dataset/weibo/weibo_files/new_test_enity.pickle', 'rb')
    test_enitys = pickle.load(test_vocab, encoding='latin1')

    train_object=torch.Tensor(train_object)
    dev_object=torch.Tensor(dev_object)
    test_object=torch.Tensor(test_object)

    X_train_tid, X_train, y_train, word_embeddings, adj = pickle.load(open(pre + "/train.pkl", 'rb'))
    X_dev_tid, X_dev, y_dev = pickle.load(open(pre + "/dev.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open(pre + "/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings
    config['node_embedding'] = pickle.load(open(pre + "/node_embedding.pkl", 'rb'))[0]
    print("#nodes: ", adj.shape[0])

    with open(pre+ '/new_id_dic.json', 'r') as f:
        newid2mid = json.load(f)
        newid2mid = dict(zip(newid2mid.values(), newid2mid.keys()))
    mid2num = {}
    for file in os.listdir(os.path.dirname(os.getcwd())+'/dataset/weibo/weibocontentwithimage/original-microblog/'):
        mid2num[file.split('_')[-2]] = file.split('_')[0]
    newid2num = {}
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    config['newid2imgnum'] = newid2num

    return X_train_tid, X_train, y_train, \
           X_dev_tid, X_dev, y_dev, \
           X_test_tid, X_test, y_test, adj,train_content,dev_content,test_content,train_object,dev_object,test_object,train_enitys,dev_enitys,test_enitys

def load_original_adj(adj):
    pre = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_files/'
    path = os.path.join(pre, 'original_adj')
    with open(path, 'r') as f:
        original_adj_dict = json.load(f)
    original_adj = np.zeros(shape=adj.shape)
    for i, v in original_adj_dict.items():
        v = [int(e) for e in v]
        original_adj[int(i), v] = 1
    return original_adj

def train_and_test(model):
    model_suffix = model.__name__.lower().strip("text")
    res_dir = 'exp_result'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.task)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.description)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = config['save_path'] = os.path.join(res_dir, 'best_model_in_each_config')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    config['save_path'] = os.path.join(res_dir, args.thread_name + '_' + 'config' + args.config_name.split(".")[
        0] + '_best_model_weights_' + model_suffix)
    #
    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))

    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, adj,train_content,dev_content,test_content,train_object,dev_object,test_object,train_enitys,dev_enitys,test_enitys= load_dataset()
    original_adj = load_original_adj(adj)
    nn = model(config, adj, original_adj)

    nn.fit(X_train_tid, X_train, y_train,
           X_dev_tid, X_dev, y_dev,train_content,dev_content,train_object,dev_object,train_enitys,dev_enitys)

    y_pred = nn.predict(X_test_tid, X_test,y_test,test_content,test_object,test_enitys)
    res = classification_report(y_test, y_pred, target_names=config['target_names'], digits=3, output_dict=True)
    for k, v in res.items():
        print(k, v)
    print("result:{:.4f}".format(res['accuracy']))
    res2={}
    res_final = {}
    res_final.update(res)
    res_final.update(res2)
    return res

config = process_config(config_file.config)
#seed = config['seed']
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)
#random.seed(seed)
model = MFAN
train_and_test(model)












