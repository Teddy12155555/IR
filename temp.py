from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, namedtuple
import re
import collections
import numpy as np
import math
from tqdm import tqdm
import numba as nb
from numba import jit
import pickle


def loadDocuments(dlist):
    FOLDER_PATH = "./docs/"
    TF = {}
    IDF = {}
    df = {}
    documents = []
    allLen = len(dlist)
    index = 0
    for docId in tqdm(dlist):
        f = open(FOLDER_PATH + docId + ".txt")
        words = f.read().split()
        # save doc data
        documents.append(words)
        TFtemp = {}
        for word in words:
            TFtemp[word] = TFtemp.get(word, 0) + 1
        # calculate IDF
        for word in TFtemp.keys():
            df[word] = df.get(word, 0) + 1
        for k, v in df.items():
            IDF[k] = math.log(allLen - v + 0.5) - math.log(v + 0.5)

        TF[str(index)] = TFtemp
        index += 1

    return TF, IDF


def loadQueries(qlist):
    FOLDER_PATH = "./queries/"
    queries = []
    for qId in tqdm(qlist):
        f = open(FOLDER_PATH + qId + ".txt")
        words = f.read().split()
        queries.append(words)

    return queries


def sim_all(self, querys):
    scores = []
    for index in range(30000):
        score = self.sim(querys, index)
        scores.append(score)

    return scores

@jit
def similarity(vec1,vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 * norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


@jit
def TFIDF(tf,idf):
    return (1+math.log(tf)) * (idf+1)


if __name__ == "__main__":
    # read query
    f = open("./query_list.txt", "r")
    queryNumList = []
    line = f.readline()
    while line:
        line = line.replace("\n", "")
        queryNumList.append(line)
        line = f.readline()
    f.close()

    # read document
    f = open("./doc_list.txt", "r")
    docNumList = []
    line = f.readline()
    while line:
        line = line.replace("\n", "")
        docNumList.append(line)
        line = f.readline()
    f.close()

    # TF, IDF = loadDocuments(docNumList)
    queries = loadQueries(queryNumList)

    with open("./TF.pickle", "rb") as f:
        TF = pickle.load(f)
    with open("./IDF.pickle", "rb") as f:
        IDF = pickle.load(f)

    # output
    outputFile = open("./hw5_result.txt", "w")
    outputFile.write("Query,RetrievedDocuments\n")
    allWordsLen = len(IDF.keys())
    allDocsLen = len(TF.keys())
    # construct index-word
    WordIndex = {}
    index = 0
    for key,value in IDF.items():
        WordIndex[key] = str(index)
        index += 1


    for index in tqdm(range(len(queries))):
        nowQuery = queries[index]
        outputFile.write(str(queryNumList[index]))
        sim = np.zeros(allDocsLen)
        # query vectory
        queryVec = np.zeros(allWordsLen)
        for q in nowQuery:
            # (TF = 1) * IDF
            queryVec[int(WordIndex[q])] = 1 * IDF[q]

        
        # document vectory
        progress = 0
        for doc,docTF in TF.items():
            print(f"At Query {index+1} Progress : {progress}/{allDocsLen}\r",end="")
            docVec = np.zeros(allWordsLen)
            for word,tf_ in docTF.items():
                docVec[int(WordIndex[word])] =   (1 + math.log(tf_)) * (IDF[word]+1)
            
            cos = similarity(docVec,queryVec)
            sim[progress] = cos
            progress += 1
        """
        
        for docIndex in range(allDocsLen):
            print(f"At Query {index+1} Progress : {docIndex}/{allDocsLen}\r",end="")
            
            docVec = np.zeros(allWordsLen)
            for word in TF[str(docIndex)].keys():
                
                docVec[int(WordIndex[word])] =  (1 + math.log(TF[str(docIndex)][word])) * IDF[word]
        """
        
        zipped = zip(docNumList, sim)
        zipped = sorted(zipped, key=lambda t: t[1], reverse=True)
        sortedDoc, score = zip(*zipped)
        for i in range(len(sortedDoc)):
            if i == 0:
                outputFile.write("," + str(sortedDoc[i]))
            else:
                outputFile.write(" " + str(sortedDoc[i]))
        outputFile.write("\n")
        
    """
    tfpath = open("./TF.pickle", "wb")
    pickle.dump(TF, tfpath)
    idfpath = open("./IDF.pickle", "wb")
    pickle.dump(IDF, idfpath)
    """

    """
    outputFile = open("./hw5_result.txt", "w")
    outputFile.write("Query,RetrievedDocuments\n")
    for index in range(len(queries)):
        nowQuery = queries[index]
        outputFile.write(str(queryNumList[index]))
        # cosine
    """
