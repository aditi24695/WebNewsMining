import csv
import snowballstemmer
import re
import math
import numpy as np

from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
stemmer = snowballstemmer.stemmer('english');
scrapper_link = []
description = ''
dictionary_vocab = {}
dictionary_df = {}
tfidf = {}
N = 30
stopword = ['ourselves' 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

class readFile():
    def fileRead(self):
        prePeocess = Preprocessing()
        with open('sample.csv', 'r+') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                description = row[0]
                tag = row[1]
                flag = row[2]
                tokens = prePeocess.tokenize(description)
                tokens = prePeocess.stopword_elimination(tokens)
                tokens = prePeocess.stemming(tokens)
                prePeocess.storeDict(tokens)
                prePeocess.storeDicdf(tokens)
                #prePeocess.calcdf(tokens)
            prePeocess.calcIdf()
            prePeocess.printIdf()

    def calcN(self):
        count = 0
        with open('sample.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                count = count + 1
        return count

class Preprocessing():
    def tokenize(self, text):
        if(text == ''): return
        tokens = text.replace("\n", '').split(" ");
        tokens =list(filter(None, tokens))
        tokens = [re.sub('[^A-Za-z]+', '', token) for token in tokens]
        tokens = [token.lower() for token in tokens if not token == '' and len(token) > 1]
        return tokens

    # Stop word removal - Standaed nltk library
    def stopword_elimination(self, tokens):
        tokens = [token for token in tokens if token not in stopword]
        return tokens


    #Snowball stemmer
    def stemming(self, tokens):
        tokens = [stemmer.stemWord(token) for token in tokens]
        return tokens

    def storeDict(self, tokens):
        global dictionary_vocab
        for token in tokens:
            if token in dictionary_vocab:
                dictionary_vocab[token] += 1
            else:
                dictionary_vocab[token] = 1


    def storeDicdf(self, tokens):
        global dictionary_df
        u_token = set(tokens)
        for token in u_token:

            if token in dictionary_df:
                dictionary_df[token] += 1
            else:
                dictionary_df[token] = 1
        return


    def calcIdf(self):
        global dictionary_df
        global tfidf
        tfidf = {}
        for i in dictionary_df:
            if dictionary_df[i] == 0: pass
            else:

                tfidf[i] = N/dictionary_df[i]

    def printIdf(self):
        global tfidf
        print(tfidf)

def main():
    rf = readFile()
    rf.fileRead()

if(__name__=='__main__'):
    main()

