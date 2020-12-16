import re
import collections
import numpy as np
import pandas as pd
import math
import time
import utils
from tqdm import tqdm
from numba import jit
import pickle


class BM25(object):
    def __init__(self, docs):
        self.docs_len = len(docs)
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.docs_len
        self.docs = docs
        self.f = []
        self.df = {}
        self.idf = {}
        #
        self.k1 = 0.8
        self.b = 0.7
        self.init()

    def init(self):
        print("init ...")
        for doc in tqdm(self.docs):
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1
            self.f.append(tmp)
            for key in tmp.keys():
                self.df[key] = self.df.get(key, 0) + 1

            for key, value in self.df.items():
                self.idf[key] = math.log(self.docs_len - value + 0.5) - math.log(
                    value + 0.5
                )

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (
                self.idf[word]
                * self.f[index][word]
                * (self.k1 + 1)
                / (
                    self.f[index][word]
                    + self.k1 * (1 - self.b + self.b * d / self.avgdl)
                )
            )
        return score

    def sim_all(self, doc):
        scores = np.array([],dtype="float16")
        for index in range(self.docs_len):
            score = self.sim(doc, index)
            scores = np.append(scores,score)

        return scores


def dataLoader(file_List, FOLDER_PATH):
    all_list = []
    words = []
    for id_ in file_List:
        f = open(FOLDER_PATH + id_ + ".txt", "r")
        text = f.read()
        words = re.findall(r"\w+", text)
        f.close()
        all_list.append(words)

    return all_list


if __name__ == "__main__":
    # read docs
    f = open("./doc_list.txt", "r")
    docList = []
    line = f.readline()
    while line:
        line = line.replace("\n", "")
        docList.append(line)
        line = f.readline()
    f.close()

    # read querys
    f = open("./query_list.txt", "r")
    queryList = []
    line = f.readline()
    while line:
        line = line.replace("\n", "")
        queryList.append(line)
        line = f.readline()
    f.close()

    docs = dataLoader(docList, "./docs/")
    queries = dataLoader(queryList, "./queries/")

    outputFile = open("./bm25_result.txt", "w")
    outputFile.write("Query,RetrievedDocuments\n")

    start = time.time()
    s = BM25(docs)
    tovsm = {}
    for i in tqdm(range(len(queryList))):
        outputFile.write(str(queryList[i]))
        sim_score = s.sim_all(queries[i])
        
        tovsm[str(queryList[i])] = sim_score
        continue

        reslut = sim_score
        zipped = zip(docList, reslut)
        zipped = sorted(zipped, key=lambda t: t[1], reverse=True)
        sortedDoc, score = zip(*zipped)
        for i in range(len(sortedDoc)):
            if i == 0:
                pass
                outputFile.write("," + str(sortedDoc[i]))
            else:
                pass
                outputFile.write(" " + str(sortedDoc[i]))
        outputFile.write("\n")

    end = time.time()
    path = open("./SimScore.pickle", "wb")
    pickle.dump(tovsm, path)

    print(f"Time Use : {end-start} seconds")
