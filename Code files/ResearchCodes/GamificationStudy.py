# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 12:37:58 2022

@author: vkhade
"""

#%%
from tkinter import font
import pandas as pd
import nltk
import numpy as np
#import string
#from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Reading excel file
df = pd.read_excel(r'Org_GamificationData.xlsx', sheet_name='MSF')
u_id = pd.DataFrame(df, columns=['User ID'])
reqdes_list = pd.DataFrame(df, columns=['Description'])
reqdes_list = reqdes_list.values.tolist()
reqlo = []
#req = reqdes_list[0][0].lower()
#print(req)
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
lemm = WordNetLemmatizer()
stop_wrds = ['a', 'the', 'an', 'that', 'this', 'of', 'i',
 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
 "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
 "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'if', 'as', 'but', 'while','to','or','for',
 'in' ,'be']
# Preprocessing of Data - Lowercase & Tokenize

for i in range(len(reqdes_list)) :
    #print(req)
    req = tokenizer.tokenize(reqdes_list[i][0].lower())
    req = [w for w in req if w not in stop_wrds]
    #req = [w for w in req if w not in stopwords.words('english')]
    req = ([stemmer.stem(q) for q in req])
    #req = [lemm.lemmatize(i) for i in req]
    reqlo.append(req)
print(reqlo)

# %% Novelty
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import linear_kernel

#d_vectors = TfidfVectorizer().fit_transform([reqlo[20]] + reqlo)

#cos_sim = linear_kernel(d_vectors[0:1], d_vectors).flatten()
#d_scores = [item.item() for item in cos_sim[1:]]
#print(d_scores)

# %% Novelty using GloVe
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity

glove = api.load("glove-wiki-gigaword-50")    
# %% Similarity calculation for MSF (Non-gamified) data

similarity_index = WordEmbeddingSimilarityIndex(glove)

dictionary = Dictionary(reqlo)
tf = TfidfModel(dictionary=dictionary)

sim_mat = SparseTermSimilarityMatrix(similarity_index, dictionary, tf)

#q_tf = tf[dictionary.doc2bow(reqlo[0])]

#index = SoftCosineSimilarity(tf[[dictionary.doc2bow(d) for d in reqlo]], sim_mat)
#d_simscores = index[q_tf]
d_simmat = []

#sort_index = np.argsort(d_simscores)[::-1]
#for i in sort_index:
    #print(f'{i} \t {d_simscores[i]:0.3f} \t {reqdes_list[i]}')

for req in reqlo:
    q_tf = tf[dictionary.doc2bow(req)]
    index = SoftCosineSimilarity(tf[[dictionary.doc2bow(d) for d in reqlo]], sim_mat)
    d_simscores = index[q_tf]
    d_simmat.append(d_simscores)

d_simmat = np.stack(d_simmat)

# %% Similarity Heatmap - MSF (Non-gamified) 
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

plt.figure(figsize=(8, 6), 
           dpi = 600)
ax = sns.heatmap(d_simmat, linewidth=0.5, vmax=0.6)
plt.xlabel('Requirements')
plt.ylabel('Requirements')
plt.show()

#%% MSF participants

u_id = u_id.values.tolist()
#print(u_id)
uid = []
c = 1
tally = []

for i in range(len(u_id)):
    u = u_id[i][0].split("_")
    u = int(u[1][:])
    uid.append(u)

for i in range(1, 15):
    count = 0
    for v in uid:
        if v == i:
            count += 1
    tally.append(count)
t = sum(tally)
tally = np.array(tally)
avgs = tally/t
#%%
plt.plot(tally)
plt.axhline(y = t/len(tally), color = 'r', linestyle = '--')

#%% Gamified data

# Reading excel file for Pointagram data
dfp = pd.read_excel(r'C:\Gamification\Org_GamificationData.xlsx', sheet_name='PG')
u_idp = pd.DataFrame(dfp, columns=['User ID'])
req_pg = pd.DataFrame(dfp, columns=['Description'])
req_pg = req_pg.values.tolist()
r_pg = []

for i in range(len(req_pg)) :
    #print(req)
    req = tokenizer.tokenize(req_pg[i][0].lower())
    req = [w for w in req if w not in stop_wrds]
    #req = [w for w in req if w not in stopwords.words('english')]
    req = ([stemmer.stem(q) for q in req])
    #req = [lemm.lemmatize(i) for i in req]
    r_pg.append(req)
print(r_pg)

#%% PG participants

u_idp = u_idp.values.tolist()
#print(u_idp)
uidp = []
c_p = 1
tally_p = []

for i in range(len(u_idp)):
    up = u_idp[i][0].split("_")
    up = int(up[1][:])
    uidp.append(up)
#print(uidp)
for i in range(1, 12):
    count = 0
    for v in uidp:
        if v == i:
            count += 1
    tally_p.append(count)
t_p = sum(tally_p)
tally_p = np.array(tally_p)
avgs_p = tally_p/t_p

#%% Comparing requirements per participant b/w MSF(Non-gamified) & PG(Gamified)



x_tal = np.arange(len(tally))
x_talp = np.arange(len(tally_p))
xtick = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
plt.figure(figsize=(15, 10))#, dpi=1000)
plt.plot(tally, color = 'purple', linewidth = 3)
plt.axhline(y = t/len(tally), color = 'm', linestyle = '--')
plt.bar(x_tal - 0.2, tally, 0.4, label = 'Non-gamified', color = 'purple')
plt.plot(tally_p, color = 'orange', linewidth = 3)
plt.axhline(y = t_p/len(tally_p), color = 'orange', linestyle = '--')
plt.bar(x_talp + 0.2, tally_p, 0.4, label = 'Gamified', color = 'orange')
plt.xlabel('Participant teams',  fontsize=20)
plt.ylabel('Requirements per team',  fontsize=20)
plt.xticks(np.arange(len(tally)),xtick, fontsize= 14)
plt.yticks(fontsize= 14)
plt.legend(loc="upper right",  fontsize=16)
plt.show()

#var_labels = sheet1.columns
#var_labels = var_labels[1:].values.tolist()
#varscore_msf = vmsf_vin.sum(axis = 0)
#varscore_pg = pg_vin.sum(axis = 0)


#plt.figure(figsize = (24, 10), dpi = 600)



#plt.xticks(x_var, var_labels)
#plt.xticks(fontsize= 14)#, fontweight= "bold")
#plt.yticks(fontsize= 14)
#plt.margins(x=0, y=0)

#plt.xlabel("Categories", fontsize = 16, labelpad=3)
#plt.ylabel("Count", fontsize = 16)
#plt.legend(fontsize= 16)
#plt.show()
#%% Similarity calculation for PG (Gamified) data

#similarity_index = WordEmbeddingSimilarityIndex(glove)

pg_dictionary = Dictionary(r_pg)
pg_tf = TfidfModel(dictionary=pg_dictionary)

pg_sim_mat = SparseTermSimilarityMatrix(similarity_index, pg_dictionary, pg_tf)

#q_tf = tf[dictionary.doc2bow(reqlo[0])]

#index = SoftCosineSimilarity(tf[[dictionary.doc2bow(d) for d in reqlo]], sim_mat)
#d_simscores = index[q_tf]
pgd_simmat = []

#sort_index = np.argsort(d_simscores)[::-1]
#for i in sort_index:
    #print(f'{i} \t {d_simscores[i]:0.3f} \t {reqdes_list[i]}')

for req in r_pg:
    pgq_tf = pg_tf[pg_dictionary.doc2bow(req)]
    index = SoftCosineSimilarity(pg_tf[[pg_dictionary.doc2bow(d) for d in r_pg]], pg_sim_mat)
    pgd_simscores = index[pgq_tf]
    pgd_simmat.append(pgd_simscores)

pgd_simmat = np.stack(pgd_simmat)

#%% Similarity Heatmap - PG (Gamified)

plt.figure(figsize=(8, 6), 
           dpi = 600)
ax = sns.heatmap(pgd_simmat, linewidth=0.5, vmax=0.6)
plt.xlabel('Requirements')
plt.ylabel('Requirements')
plt.show()



#%% Setting threshold to 0.6 and calculating count & sum of values below threshold

th = 0.6
l_msf = 77
sum_msf = []
count_msf = []
sbc_msf = []
sbc_msfn = []
for j in range(l_msf):
    s = 0
    cnt = 0
    i = 0
    for i in range(l_msf):
        val = d_simmat[i,j]
        if val < 0.6:
            s += val
            cnt += 1
    sum_msf.append(s)
    count_msf.append(cnt)
    sbc_msf.append(s/cnt)

for i in range(len(sbc_msf)):
    val = sbc_msf[i]
    if val < 0.3 and count_msf[i] > 70:
        sbc_msfn.append([i, val])
#plt.scatter(sum_msf, count_msf)
#%% Setting threshold to 0.6 and calculating count & sum of values below threshold

th = 0.6
l_pg = 109
sum_pg = []
count_pg = []
sbc_pg = []
sbc_pgn = []

for j in range(l_pg):
    s = 0
    cnt = 0
    i = 0
    for i in range(l_pg):
        val = pgd_simmat[i,j]
        if val < 0.6:
            s += val
            cnt += 1
    sum_pg.append(s)
    count_pg.append(cnt)
    sbc_pg.append(s/cnt)
    
for i in range(len(sbc_pg)):
    val = sbc_pg[i]
    if val < 0.3 and count_pg[i] > 102:
        sbc_pgn.append([i, val])

#plt.scatter(sum_pg, count_pg)
#%%
plt.plot(sbc_msf)
plt.plot(sbc_pg)

#%% calculating variance in participation rate
#import statistics

print("Variance of non-gamified data set is % s" %(np.var(tally)))
print("Standard deviation of non-gamified data set is % s" %(np.std(tally)))
print("Variance of non-gamified data set is % s" %(np.var(tally_p)))
print("Standard deviation of non-gamified data set is % s" %(np.std(tally_p)))

#%% checking if quantity(tally- requirements/participant) data is normally distributed
import scipy
from scipy import stats
s, pval = scipy.stats.normaltest(tally, axis=0, nan_policy='propagate')
s_g, pval_g = scipy.stats.normaltest(tally_p, axis=0, nan_policy='propagate')

#%% Signed rank test
#from scipy.stats import wilcoxon
#wq, p_valq = wilcoxon(tally, tally_p)
s_ind, pval_ind = stats.ttest_ind(tally, tally_p)

print(s_ind)
print(pval_ind)
#%% Variety MSF(Non-gamified)


import pandas as pd

#Reading two Excel Sheets

sheet1 = pd.read_excel(r'C:\Gamification\Org_GamificationDataV_R1.xlsx', sheet_name='Variety_MSFV')
sheet2 = pd.read_excel(r'C:\Gamification\Org_GamificationDataM_R3.xlsx', sheet_name='Variety_MSFV')

# Iterating the Columns Names of both Sheets
for i,j in zip(sheet1,sheet2):
	
	# Creating empty lists to append the columns values	
	a,b =[],[]

	# Iterating the columns values
	for m, n in zip(sheet1[i],sheet2[j]):

		# Appending values in lists
		a.append(m)
		b.append(n)

	# Sorting the lists
	a.sort()
	b.sort()

	# Iterating the list's values and comparing them
	for m, n in zip(range(len(a)), range(len(b))):
		if a[m] != b[n]:
			print('Column name : \'{}\' and Row Number : {}'.format(i,m))
#%% 
vmsf_vin = sheet1.to_numpy()
vmsf_vin = vmsf_vin[:, 1:]
#print(vmsf_vin)

vmsf_mo = sheet2.to_numpy()
vmsf_mo = vmsf_mo[:, 1:]
irr = 1 -((vmsf_mo != vmsf_vin).sum()/float(vmsf_vin.size))
print(irr)
#%% Cohen's kappa to calculate Interrater reliability for MSF (Non-gamified)
import numpy as np
from sklearn.metrics import cohen_kappa_score

vmsf_vin = vmsf_vin.astype(float)
vmsf_mo = vmsf_mo.astype(float)
k_score_msf = []
for i in range(len(vmsf_vin)):
    k = cohen_kappa_score(vmsf_vin[i, :], vmsf_mo[i, :])
    k_score_msf.append(k)
k_score_msf = np.asarray(k_score_msf)
print(np.mean(k_score_msf))
#%% Variety MSF(Gamified)

import pandas as pd

#Reading two Excel Sheets

sheet1_pg = pd.read_excel(r'C:\Gamification\Org_GamificationDataV_R1.xlsx', sheet_name='Variety_PG')
sheet2_pg = pd.read_excel(r'C:\Gamification\Org_GamificationDataM_R3.xlsx', sheet_name='Variety_PG')

# Iterating the Columns Names of both Sheets
for i,j in zip(sheet1_pg,sheet2_pg):
	
	# Creating empty lists to append the columns values	
	a,b =[],[]

	# Iterating the columns values
	for m, n in zip(sheet1_pg[i],sheet2_pg[j]):

		# Appending values in lists
		a.append(m)
		b.append(n)

	# Sorting the lists
	a.sort()
	b.sort()

	# Iterating the list's values and comparing them
	for m, n in zip(range(len(a)), range(len(b))):
		if a[m] != b[n]:
			print('Column name : \'{}\' and Row Number : {}'.format(i,m))
#%% 
pg_vin = sheet1_pg.to_numpy()
pg_vin = pg_vin[:, 1:]
#print(vmsf_vin)

pg_mo = sheet2_pg.to_numpy()
pg_mo = pg_mo[:, 1:]
irr = 1 -((pg_mo != pg_vin).sum()/float(pg_vin.size))
print(irr)
#%% Cohen's kappa to calculate Interrater reliability for MSF (Non-gamified)
import numpy as np
from sklearn.metrics import cohen_kappa_score

pg_vin = pg_vin.astype(float)
pg_mo = pg_mo.astype(float)
k_score_pg = []
for i in range(len(pg_vin)):
    k = cohen_kappa_score(pg_vin[i, :], pg_mo[i, :])
    k_score_pg.append(k)
k_score_pg = np.asarray(k_score_pg)
print(np.mean(k_score_pg))

#%% Plotting bar plots for Variety comparison
import matplotlib.pyplot as plt


var_labels = sheet1.columns
var_labels = var_labels[1:].values.tolist()
varscore_msf = vmsf_vin.sum(axis = 0)
varscore_pg = pg_vin.sum(axis = 0)
x_var = np.arange(len(var_labels))

plt.figure(figsize = (24, 10), dpi = 600)
plt.bar(x_var - 0.2, varscore_msf, 0.4, label = 'Non-gamified')
plt.bar(x_var + 0.2, varscore_pg, 0.4, label = 'Gamified')

plt.xticks(x_var, var_labels)
plt.xticks(fontsize= 14)#, fontweight= "bold")
plt.yticks(fontsize= 14)
plt.margins(x=0, y=0)

plt.xlabel("Categories", fontsize = 16, labelpad=3)
plt.ylabel("Count", fontsize = 16)
plt.legend(fontsize= 16)
plt.show()

#plt.savefig()

#%% normality of variety tally
s_msfv, pval_msfv = scipy.stats.normaltest(varscore_msf, axis=0, nan_policy='propagate')
print(s_msfv, pval_msfv)
s_pgv, pval_pgv = scipy.stats.normaltest(varscore_pg, axis=0, nan_policy='propagate')
print(s_pgv, pval_pgv)

#%% mann-whitney test - Statistical Inference test
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

tstat_var, pval_var = mannwhitneyu(varscore_msf, varscore_pg)
print(tstat_var, pval_var)

# Chi-square contingency test for variety
obs = np.stack([varscore_msf, varscore_pg])
#obs = np.transpose(obs)
vcon_chstat, vcon_pval, d, ex = chi2_contingency(obs)
print(vcon_chstat, vcon_pval)

# Completeness proportions test
comp_count = np.array([69, 81])
tot_count = np.array([77, 88])
tstat_comp, pval_comp = proportions_ztest(comp_count, tot_count)
print(tstat_comp, pval_comp)

# Novelty proportions test
nov_count = np.array([14, 17])
totnov_count = np.array([77, 109])
tstatno_comp, pvalno_comp = proportions_ztest(nov_count, totnov_count)
print(tstatno_comp, pvalno_comp)