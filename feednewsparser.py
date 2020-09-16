from __future__ import division
from itertools import groupby
import feedparser
import sqlite3 as sql
import collections
from collections import defaultdict
import urllib.request as urllib2
from bs4 import BeautifulSoup as bs
import vector
import numpy as np
import nltk
import re
import snowballstemmer
import math
import sys

from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

stemmer = snowballstemmer.stemmer('english');


link_scrap = []
keywords = []
scrapper_link = []
description = ''
dictionary_vocab = {}
dictionary_df = {}
N = 0
stopword = ['ourselves' 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

### prototype : potential, tfidf, category,support

class Scrapper():
    # Creates Database Connection
    def create_connection(self):
        database = 'C:\\Users\\aditi\\OOAD\\db.sqlite3'
        return sql.connect(database)


# Go to the topic page of the timesofIndia and filter out all the topics and their respective RSS links
    def scrap_rss_page(self):
        page = 'https://timesofindia.indiatimes.com/rss.cms'
        page_content = urllib2.urlopen(page)
        soup = bs(page_content, "html.parser")
        div_content = soup.find_all("div", id='main-copy')
        count = 0
        for tag in div_content:
            tdtags = tag.find_all("td")
            for tag in tdtags:
                if tag.select('.rssurl'):
                    for link in tag.select('.rssurl'):
                        scrapper_link.append(link['href'])
                if tag.text:
                    if tag.text != '' and tag.text != 'More':
                        keywords.append(tag.text)

        for i in range(len(keywords)):
            keywords_list = ['Top Stories', 'India', 'World', 'Business', 'Sports', 'Health', 'Science', 'Environment','Education']
            if (keywords[i] in keywords_list):
                self.parse_rss(scrapper_link[i], keywords[i])


# parse rss_link of each page and save the required xml tags in the list news details
    def parse_rss(self,link, tag):
        preprocess = Preprocessing()
        classify = Classification()
        d = feedparser.parse(link)
        news_details = []
        for post in d['entries']:
            link = (post.link)
            description = self.beautiful_soup(link)
            #news_details.append([post.title, description, post.link, post.published, tag, 0])
            global N
            N = N+1
            if N > 15: classify.clasifyRule(text=description, category=tag)
            preprocess.tokenize(description, tag)
        #self.store(news_details);


# from the news_details list store the required topics from xml in the db
    def store(self,news_details):
        con = self.create_connection();
        c = con.cursor()
        c.executemany('INSERT INTO newsapp_topstories (title,desc,link,published,tag,status) VALUES (?,?,?,?,?,?);',
                      news_details)

        con.commit()
        con.close()

    def beautiful_soup(self, each_link):
        page = each_link
        content=""
        try:
            page_content = urllib2.urlopen(page)
        except Exception as e:
            print(page, "Something Bad Happened !")
            return ""
        else:
            soup = bs(page_content, "html.parser")
            content = ""
            div_content = soup.find_all('div', {'class': "article_content clearfix"})
            for wrapper in div_content:
                content = wrapper.text
        return content


    def main(self):
        self.scrap_rss_page()

class Preprocessing():
    def tokenize(self, text, category):
        if(text == ''): return
        tokens = text.replace("\n", '').split(" ");
        tokens =list(filter(None, tokens))
        tokens = [re.sub('[^A-Za-z0-9]+', '', token) for token in tokens ]
        tokens = [token.lower() for token in tokens]
        if(category == ""): return tokens
        self.stopword_elimination(tokens, category)

    # Stop word removal - Standaed nltk library
    def stopword_elimination(self, tokens,category):
        tokens = [token for token in tokens if token not in stopword]
        if(category == ""): return tokens
        self.stemming(tokens, category)

    #Snowball stemmer
    def stemming(self, tokens, category):
        tokens = [stemmer.stemWord(token) for token in tokens]
        if(category == ""): return tokens
        self.storeDicdf(tokens)
        self.storeDict(tokens,category)

    def storeDict(self, tokens, category):
        tf_idf = C_tfidf()
        for token in tokens:
            if token in dictionary_vocab:
                dictionary_vocab[token] += 1
            else:
                dictionary_vocab[token] = 1
        if(category == ""): return ;
        tf_idf.calctf(tokens, category)

    def storeDicdf(self, tokens):
        u_token = set(tokens)
        for token in u_token:
            if token in dictionary_df:
                dictionary_df[token] += 1
            else:
                dictionary_df[token] = 1
        return

class C_tfidf():
    def calctf(self, tokens, category):
        v1 = []
        temp_dict = dictionary_vocab.copy()
        for key in temp_dict:
            temp_dict[key] = 0
        for token in tokens:
            temp_dict[token] = temp_dict[token]+ 1
        for i in temp_dict:
            v1.append(temp_dict[i])
        if (category == ""): return v1
        self.calctfidf(v1,category)

    def calctfidf(self, v1, category):
        potentialc = PotentialCalculation()
        global N
        tfidf = []
        index = 0
        v2 = np.asarray(dictionary_vocab.values())
        for i in dictionary_vocab:
            temp_v = N/dictionary_df[i]
            tfidf.append(v1[index]*math.log(1+temp_v, 10))
            index = index+1
        k = N
        tfidf = self.prune_tfidf(tfidf)
        if(category == ""):return tfidf
        potentialc.calculatePotential(tfidf, k, category)
        return tfidf

    def prune_tfidf(self, v1):
        for i in range(0, len(v1)):
            if i < 1.5:
                v1[i] = 0
        return v1


class PotentialCalculation():
    allProto = []
    prev_b_value = []
    prev_sigma_value = 0

    def calculatePotential(self, tfidf,k, category):
        A = {}
        A['category']= category
        A['tfidf'] = tfidf
        if(k == 1):
            A['potential'] = 1
            self.insertPrototype(A)
            return 1
        n = len(tfidf)
        potential = 1/(2-(self.valueB(tfidf,k, A)/((k-1)*self.norm(A['tfidf']))))
        A['potential'] = potential
        self.updatePrototype(A)
        return potential

    def norm(self, tfidf):
        return self.sqrt(self.calcsq(tfidf))

    def valueB(self, tfidf, k, A):
        b_v =self.calcdotProd(tfidf, self.value_b(tfidf, k, A))
        return b_v

    def value_b(self, tfidf, k, A ):
        b_value = []
        if(len(self.prev_b_value)==0): self.prev_b_value = [0]*len(tfidf)
        for i in range(0, len(tfidf)):
            try:
                a = self.local_sq_sum_ele(A, i)
                b = self.sqrt((tfidf[i]**2)/a)
                c = self.prev_b_value[i] + b
                b_value.append(c)
            except:
                b_value.append(0)
        self.prev_b_value = b_value
        return b_value


    def calcQ(self, A):
        support = 0
        for prototype in self.allProto:
            if(A['category'] == prototype['category']):
                support = support + prototype['support']
        return support

    def update_support(self, A):
        min_dist = sys.maxsize
        temp_proto = {}
        print('len',len(self.allProto))

        for prototype in self.allProto:
            if prototype['category'] == A['category']:
                dist_list = [x - y for x,y in zip(A['tfidf'], prototype['tfidf'])]
                dist = self.norm(dist_list)
                if dist<min_dist:
                    min_dist = dist
                    temp_proto = prototype

            print('pro supp',prototype['support'],'pro pot',prototype['potential'], prototype['category'])
        for prototype in self.allProto:
            if prototype == temp_proto:
                prototype['support'] += 1
                print('in support',prototype['support'])

    def local_sq_sum(self, A):
        sum = 0
        for prototype in self.allProto:
            if(prototype['category'] == A['category']):
                sum = sum + self.calcsq(prototype)
        return sum

    def local_sq_sum_ele(self, A, j):
        sum = 0
        for prototype in self.allProto:
            if prototype['category'] == A['category']:
                sum = sum + (prototype['tfidf'][j])**2
        return sum

    def updatePrototype(self, A):
        for prototype in self.allProto:
            try:
                if prototype['category'] == A['category']:
                    print('category A',A['category'],'cate pro',prototype['category'])
                    print('calQ-1',(self.calcQ(A) - 1))
                    print('potential',prototype['potential'])
                    print(1/(prototype['potential'])-1)
                    print(self.calcCosDist(A['tfidf'],prototype['tfidf']))
                    prototype['potential'] = ((self.calcQ(A) - 1)*prototype['potential'])/((self.calcQ(A)-1)+((self.calcQ(A)-2)*(1/(prototype['potential'])-1))+(self.calcCosDist(A['tfidf'],prototype['tfidf'])))
                    print('new pote',prototype['potential'])
            except:
                prototype['potential'] = 1
        self.comparePotential(A)
        return

    def insertPrototype(self, protoNew):
        protoNew['support'] = 1
        protoNew['potential'] = 1
        self.allProto.append(protoNew)
        self.removePrototype(A=protoNew)

    def comparePotential(self, A):
        flag = True
        for prototype in self.allProto:
            print("In compare Potential", prototype['potential'], A['potential'])
            if prototype['potential']>A['potential'] and prototype['category'] == A['category']:
                flag = False
        if(flag):
            print("Inserted")
            self.insertPrototype(A)
        else:
            print("Not Inserted:)")
            self.update_support(A)

    def removePrototype(self, A):
        print('lenght all proto',len(self.allProto))
        for prototype in self.allProto:
            if not A == prototype and A['category'] == prototype['category']:
                if self.calcMembership(A, prototype) > math.exp(-1):
                    print("Support",prototype['support'])
                    print("Removed")
                    self.allProto.remove(prototype)

    def calcMembership(self, A, prototype):
        sigma_value = self.calcSigma(A,prototype)
        if(sigma_value != 0):
            a = self.calcCosDist(A['tfidf'], prototype['tfidf'])/sigma_value
            check = math.exp((-0.5)*a)
            # print('mem',check,'cosin',self.calcCosDist(A['tfidf'], prototype['tfidf']),'sogm',sigma_value)
            return check
        else:
            return 0

    def calcSigma(self, A, prototype):
        global N
        sig = self.sqrt(self.calcsq(self.prev_sigma_value)+(self.calcsq(self.calcCosDist(A['tfidf'], prototype['tfidf']))-self.calcsq(self.prev_sigma_value))/N)
        self.prev_sigma_value = sig
        return sig

    def calcCosDist(self, Ak ,Ap):
        dist = 1 - ((self.calcdotProd(Ak, Ap))/(self.sqrt((self.calcsq(Ak))*(self.calcsq(Ap)))))
        return dist

    def calcsq(self, A):
        try:
            if int(A):
                return int(int(A)*int(A))
            else:
                return A*A
        except:
            sumsq = 0
            for i in range(len(A)):
                sumsq = sumsq + A[i]*A[i]
            return sumsq

    def sqrt(self, A):
        return A**(0.5)

    def calcdotProd(self, A, B):
        if(len(A)!= 0 and len(B)!=0):
            sum_d = sum([A[i]*B[i] for i in range(0, min(len(A), len(B)))])
        else:
            sum_d = 0
        return sum_d

class Classification():
    potentialCalc = PotentialCalculation()
    preProcess = Preprocessing()
    tfidf = C_tfidf()
    allProto = potentialCalc.allProto
    min_dist = sys.maxsize
    category = ''
    def clasifyRule(self, text, category):
        print("Category of news artiicle", category)
        print("Checking classificatioin")
        tokens = self.preProcess.tokenize(text, "")
        tokens = self.preProcess.stopword_elimination(tokens,"")
        tokens = self.preProcess.stemming(tokens, "")
        self.preProcess.storeDicdf(tokens)
        self.preProcess.storeDict(tokens, "")
        v1 = self.tfidf.calctf(tokens, "")
        new_tfidf = self.tfidf.calctfidf(v1, "")
        for protoype in self.allProto:
            dist = self.potentialCalc.calcCosDist(protoype['tfidf'], new_tfidf)
            if(self.min_dist> dist):
                self.min_dist = dist
                self.category = protoype['category']
        print("new category discovered",self.category)
        return self.category


if __name__ == '__main__':
    webscrapper = Scrapper()
    webscrapper.main()