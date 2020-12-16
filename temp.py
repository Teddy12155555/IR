
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

    # construct docid-index
    DocIndex = {}
    di = 0
    for d in docNumList:
        DocIndex[d] = str(di)
        di += 1
    
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
        zipped = zip(docNumList, sim)
        zipped = sorted(zipped, key=lambda t: t[1], reverse=True)
        sortedDoc, score = zip(*zipped)
        # ===============
        # === Rocchio ===
        # ===============
        B = 0.75
        C = 0.1
        R_time = 8
        Rq = 5
        nRq = 3
        
        resort = sortedDoc[:1500]
        for i in range(R_time):
            docVec = np.zeros(allWordsLen)
            for j in range(Rq):
                documentindex = DocIndex[str(resort[j])]
                for word,tf_ in TF[documentindex].items():
                    docVec[int(WordIndex[word])] += (B / Rq) * (1 + math.log(tf_)) * (IDF[word]+1)
        
            for j in range(nRq):
                documentindex = DocIndex[str(resort[-j])]
                for word,tf_ in TF[documentindex].items():
                    docVec[int(WordIndex[word])] -= (C / nRq) * (1 + math.log(tf_)) * (IDF[word]+1)

            newQuery = queryVec + docVec
            print("Resort...")
            newsim  = np.zeros(1500)
            progress = 0
            for resortDoc in tqdm(resort):
                temp = DocIndex[str(resortDoc)]
                docVec = np.zeros(allWordsLen)
                for word,tf_ in TF[temp].items():
                    docVec[int(WordIndex[word])] = (1 + math.log(tf_)) * (IDF[word]+1)

                cos = similarity(docVec,newQuery)
                newsim[progress] = cos
                progress += 1

            zipped2 = zip(resort, newsim)
            zipped2 = sorted(zipped2, key=lambda t: t[1], reverse=True)
            sortedDoc2, score2 = zip(*zipped2)
            resort = sortedDoc2
        

        # Write Output
        out = np.append(resort,sortedDoc[1500:])
        print(len(out))

        for i in range(len(out)):
            if i == 0:
                outputFile.write("," + str(out[i]))
            else:
                outputFile.write(" " + str(out[i]))

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
