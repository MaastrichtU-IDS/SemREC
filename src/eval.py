{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf830
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww25400\viewh15520\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \'91\'92\'92\
This is the evaluation file for subsumption task.\
The embeddings of each class are taken and the distance between those classes are calculated.\
Based on these distances, we calculate metrics like Top1, Top10, Top100, Median rank and 90th percentile rank.\
Eg. If we have class A is a subclass of class B then the distance between the embeddings of both classes must be small. \
If the distance of A is smallest among all other classes to B then it comes under Top1. Similarly if it falls under top 10 distances then \
we count it as top10. \
Search for comments which needs to be changed in order to make the code workable with your embeddings.\
\'91\'92\'92\
\
import os\
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  \
os.environ["CUDA_VISIBLE_DEVICES"] = "0"\
import click as ck\
import numpy as np\
import pandas as pd\
import logging\
import math\
import os\
from collections import deque\
import pickle as pkl\
import torch\
import torch.nn as nn\
import torch.nn.functional as F\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0
\cf0 import statistics\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0
\cf0 \
from sklearn.manifold import TSNE\
from sklearn.metrics import roc_curve, auc, matthews_corrcoef\
import matplotlib.pyplot as plt\
from scipy.stats import rankdata\
\
logging.basicConfig(level=logging.INFO)\
import operator\
from collections import Counter\
\
class Model(nn.Module):\
    \
    def __init__(self, nb_classes, embedding_size):\
        super(Model, self).__init__()\
        self.nb_classes = nb_classes\
        self.embedding_size = embedding_size\
	\
	### make changes to the embedding layer to fit the embeddings but make sure to store it in self.cls_embeddings\
        self.cls_embeddings = nn.Embedding( nb_classes, embedding_size)\
\
def load_eval_data(data_file):\
    data = []\
    rel = f'SubClassOf'\
    with open(data_file, 'r') as f:\
        for line in f:\
            it = line.strip().split()\
            id1 = it[0]\
            id2 = it[1]\
            data.append((id1, id2))\
    return data\
\
def evaluate_hits(data,cls_embeds_file, embedding_size):\
   ### make sure the embeddings are stored in a pickle file \
   ### containing total number of classes(nb_classes), the class embeddings \
   ### and a dictionary called classes_dict which maps the class names to their indices(rows) in the embedding matrix.\
		\
    with open(cls_embeds_file, 'rb') as f:\
        cls_df = pkl.load(f)\
    \
    nb_classes = len(cls_df['cls'])\
    model = Model(nb_classes, embedding_size).cuda()\
    model.load_state_dict(cls_df['embeddings'])   \
    model.eval()\
\
    embeds_list = model.cls_embeddings(torch.tensor(list(range(nb_classes))).cuda())\
\
    classes = cls_df['classes_dict\'92]\
    \
    embeds = embeds_list.detach().cpu().numpy()\
\
    size = len(embeds[0]) \
\
   ####\'97\'97\'97\'97\'97Do not make any changes to the evaluation part after this \'97\'97\'97\'97\'97\'97####	\
    \
    top1 = 0\
    top10 = 0\
    top100 = 0\
    mean_rank = 0\
    rank_vals =[]\
    for test_pts in data:\
        c = test_pts[0]\
        d = test_pts[1]\
        index_c = classes[c]\
        index_d = classes[d]\
        dist =  np.linalg.norm(embeds - embeds[index_d], axis=1) \
        dist_dict = \{i: dist[i] for i in range(0, len(dist))\} \
        s_dst = dict(sorted(dist_dict.items(), key=operator.itemgetter(1)))\
        s_dst_keys = list(s_dst.keys())\
        ranks_dict = \{ s_dst_keys[i]: i for i in range(0, len(s_dst_keys))\}\
        rank_c = ranks_dict[index_c]\
        mean_rank += rank_c\
        rank_vals.append(rank_c)\
        if rank_c == 1:\
            top1 += 1\
        if rank_c <= 10:\
            top10 += 1\
        if rank_c <= 100:\
            top100 += 1\
    \
    n = len(data)\
    top1 /= n\
    top10 /= n\
    top100 /= n\
    mean_rank /= n\
    total_classes = len(embeds)\
    return top1,top10,top100,mean_rank,rank_vals,total_classes  \
\
def compute_rank_percentile(scores,x):\
    scores.sort()\
    per = np.percentile(scores,x)\
    return per\
\
def compute_median_rank(rank_list):\
    med = np.median(rank_list)\
    return med    \
\
def calculate_percentile_1000(scores):\
    ranks_1000=[]\
    for item in scores:\
        if item < 1000:\
            ranks_1000.append(item)\
    n_1000 = len(ranks_1000)\
    nt = len(scores)\
    percentile = (n_1000/nt)*100\
    return percentile\
\
def compute_rank_roc(ranks, n):\
    auc_lst = list(ranks.keys())\
    auc_x = auc_lst[1:]\
    auc_x.sort()\
    auc_y = []\
    tpr = 0\
    sum_rank = sum(ranks.values())\
    for x in auc_x:\
        tpr += ranks[x]\
        auc_y.append(tpr / sum_rank)\
    auc_x.append(n)\
    auc_y.append(1)\
    auc = np.trapz(auc_y, auc_x)/n\
    return auc\
\
def out_results(rks_vals):\
    med_rank = compute_median_rank(rks_vals)\
    print("Median Rank:",med_rank)\
    per_rank_90 = compute_rank_percentile(rks_vals,90)\
    print("90th percentile rank:",per_rank_90)\
    percentile_below1000 = calculate_percentile_1000(rks_vals)\
    print("Percentile for below 1000:",percentile_below1000)\
    print("% Cases with rank greater than 1000:",(100 - percentile_below1000))\
\
def print_results(top1,top10,top100,mean_rank,rks_vals,n):\
    print("top1:",top1)\
    print("top10:",top10)\
    print("top100:",top100)\
    print("Mean Rank:",mean_rank)\
    rank_dicts = dict(Counter(rks_vals))\
    print("AUC:",compute_rank_roc(rank_dicts,n))\
    out_results(rks_vals) \
\
\
if __name__ == \'91__main__\'92:\
\
	###  modify test file path\
	test_file = \'91provide test file path\'92\
	test_data = load_eval_data(test_file)\
\
	### modify embedding pickle file path\
	cls_embeds_file = \'91add path.pkl'\
\
	### modify embedding size\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0
\cf0 	embedding_size = 500\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0
\cf0 \
	print('start evaluation........')\
	top1,top10,top100,mean_rank,rank_vals,n_cls = evaluate_hits(test_data,cls_embeds_file,embedding_size)\
\
\
	print("Results on test data")\
	print_results(top1,top10,top100,mean_rank,rank_vals,n_cls)\
\
\
\
\
\
\
}