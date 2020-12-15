from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, namedtuple
import re
import collections
import numpy as np
import math
from tqdm import tqdm
import numba as nb

def TF_table(col, row):
    tf_table = []
    for doc_key in tqdm(col.keys()):
        term_tf = []
        for term in row:
            term_tf.append(col[str(doc_key)][str(term)])

        tf_table.append(term_tf)

    return np.array(tf_table)

@nb.njit
def IDF_table(tf_table):
    Idf = np.array([])
    allsize = np.size(tf_table, 0)
    for col in tqdm(range(np.size(tf_table, 1))):
        count = np.count_nonzero(tf_table[:, col])
        value = math.log((1+allsize)/(count+1)) + 1
        Idf = np.append(Idf, value)

    return Idf

def loadDocuments(dlist):
    FOLDER_PATH = './docs/'
    allWords = Counter()
    TermF = {}
    for docId in tqdm(dlist):
        f = open(FOLDER_PATH + docId + '.txt')
        words = f.read().split()
        TermF[docId] = Counter(words)
        allWords.update(words)

    return np.array(allWords.keys()),TermF

def getTermFinDoc(docList):
    FOLDER_PATH = './docs/'
    TermF = {}
    for doc_id in docList:
        f = open(FOLDER_PATH+doc_id+'.txt', 'r')
        text = f.read()
        words = re.findall(r'\w+', text)
        f.close()
        count = np.array(words)
        counter = collections.Counter(count)
        TermF[str(doc_id)] = counter

    print(TermF)
    
    return TermF

if __name__ == "__main__":
    # read query
    f = open("./query_list.txt", "r")
    queryNumList = []
    line = f.readline()
    while line:
        line = line.replace('\n', '')
        queryNumList.append(line)
        line = f.readline()
    f.close()

    # read document
    f = open("./doc_list.txt", "r")
    docNumList = []
    line = f.readline()
    while line:
        line = line.replace('\n', '')
        docNumList.append(line)
        line = f.readline()
    f.close()
    
    all_,TermF = loadDocuments(docNumList)

    #DOC_TF = TF_table(TermF, all_)
    #DOC_IDF = IDF_table(DOC_TF)

    np.save('TF',DOC_TF)
    np.save('IDF',DOC_IDF)