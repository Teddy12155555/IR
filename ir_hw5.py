from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, namedtuple
import re
import numpy
import pandas as pd
import math


def TF_table(col, row):
    tf_table = []
    for doc_key in col.keys():
        term_tf = []
        for term in row:
            term_tf.append(col[str(doc_key)][str(term)])

        tf_table.append(term_tf)

    return np.array(tf_table)


def IDF_table(tf_table):
    Idf = np.array([])
    allsize = np.size(tf_table, 0)
    for col in range(np.size(tf_table, 1)):
        count = np.count_nonzero(tf_table[:, col])
        value = math.log((1+allsize)/(count+1)) + 1
        Idf = np.append(Idf, value)

    return Idf


def TF_IDF(DOC_TF, DOC_IDF):
    # must be np.array
    return DOC_TF * DOC_IDF


def dataLoader(docList):
    FOLDER_PATH = './docs/'
    docVec = []
    words = []
    for doc_id in docList:
        f = open(FOLDER_PATH+doc_id+'.txt', 'r')
        text = f.read()
        docVec.append(text)
        words = re.findall(r'\w+', text)
        f.close()
        print(len(words))
        break

    return docVec


def getAllQueryWords(qfile_list):
    query_words = []
    for q_num in qfile_list:
        path = './queries/' + q_num + '.txt'
        f = open(path, "r")
        query = re.findall(r'\w+', f.readline())
        for q in query:
            if q in query_words:
                continue
            else:
                query_words.append(q)
    return query_words


def getTermFinDoc(docList):
    FOLDER_PATH = './docs/'
    TermF = {}
    for doc_id in docList:
        f = open(FOLDER_PATH+doc_id+'.txt', 'r')
        text = f.read()
        words = re.findall(r'\w+', text)
        f.close()
        count = numpy.array(words)
        counter = collections.Counter(count)
        TermF[str(doc_id)] = counter

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
    allQueryWords = getAllQueryWords(queryNumList)

    # read document
    f = open("./doc_list.txt", "r")
    docList = []
    line = f.readline()
    while line:
        line = line.replace('\n', '')
        docList.append(line)
        line = f.readline()
    f.close()

    docs = getTermFinDoc(docList)
"""
    DOC_TF = TF_table(docs, allQueryWords)
    DOC_IDF = IDF_table(DOC_TF)

    DOC_TF_IDF = normalize(TF_IDF(DOC_TF, DOC_IDF), norm="l2")

    outputFile = open("./vsm_result.txt", "w")
    outputFile.write("Query,RetrievedDocuments\n")

    for q_num in queryNumList:
        outputFile.write(str(q_num))

        path = './queries/' + q_num + '.txt'
        f = open(path, "r")
        query = numpy.array(re.findall(r'\w+', f.readline()))

        # count TF
        counter = collections.Counter(query)
        d = dict()
        d[str(q_num)] = counter
        q_tf_table = TF_table(d, allQueryWords)
        q_idf_table = IDF_table(q_tf_table)
        QUERY_TF_IDF = normalize(
            TF_IDF(q_tf_table, q_idf_table), norm="l2")
        print(DOC_TF_IDF.shape)
        print(QUERY_TF_IDF.shape)

        result = cosine_similarity(DOC_TF_IDF, QUERY_TF_IDF)

        resList = result.tolist()
        for i in range(len(resList)):
            resList[i] = resList[i][0]

        zipped = zip(docList, resList)
        zipped = sorted(zipped, key=lambda t: t[1], reverse=True)
        sortedDoc, score = zip(*zipped)
        for i in range(len(sortedDoc)):
            if i == 0:
                outputFile.write("," + str(sortedDoc[i]))
            else:
                outputFile.write(" " + str(sortedDoc[i]))
        outputFile.write("\n")
        break
"""