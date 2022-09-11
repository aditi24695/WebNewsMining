import re
import nltk
import snowballstemmer
import math
import numpy as np
import urllib.request as urllib2
from bs4 import BeautifulSoup as bs
import sys
import csv
import copy
import feedparser
from contextlib import  contextmanager
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import tempfile
from threading import Thread,Lock
from multiprocessing import Process, Manager,freeze_support, freeze_support, Value, Array
from atomicwrites import atomic_write
import signal, os
import time
import random
import gc


stemmer = snowballstemmer.stemmer('english')
link_scrap = []
keywords = []
scrapper_link = []
dictionary_vocab = {}
dictionary_df = {}
N = 0
prev_b_value = []
description = ""
lock = Lock()
# manager = Manager()
# data_buff=manager.list()
data_buff = []
initial_buffer = []

stopword = ['ourselves' 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
k = 1
class TreeRootNode():
    def __init__(self):
        self.children = []
        self.buffer = []
    def add_child(self, A):
        C_node = CategoryNode(A)
        C_node.setParent(self)
        self.children.append(C_node)
        return C_node


root = TreeRootNode()

class CategoryNode():
    def __init__(self, A,  parent=None):
        self.category = A['category']
        self.children = []
        self.buffer = []
        self.parent = parent
        term = {}

    def setParent(self, node):
        self.parent = node

    def addTerms(self, term):
        self.term.append(term)
        pass

    def addChildren(self, A):
        P_Node = TreePrototypeNode(A)
        P_Node.setParent(self)
    #    print("A children in node", P_Node.prototype["category"])
        self.children.append(P_Node)
     #   print("Len of node", len(self.children))
        return

    def add_in_buffer(self, A, status):
        self.buffer.append(A)
        return

    def removeProto(self, p_children):
        self.children.remove(p_children)
        return

class TreePrototypeNode():
    def __init__(self, A = "", tokens = "", parent=None):
        self.children = []
        self.prototype = A
        self.potential = 1
        self.tokens = tokens
        self.parent = parent
    buffer = []
    def add_buffer(self):
        pass
    def add_prototype(self, A):
        self.children.append(A)
    def remove_children(self, prototype):
        self.children.remove(prototype)
    def setToken(self, tokens):
        self.tokens = tokens
    def setPotential(self, potential):
        self.potential = potential
    def setParent(self, node):
        self.parent = node
    def setStatus(self):
        pass


class BuildTree():
    prev_sigma_value = 0

    def createNode(self,root, A):
        global k
        flag = False
        potential = 1 #calcPotential()
        A['num'] = k
        k = k + 1
      #  print("kvalue", k)
        A['potential'] = self.calculatePotential(A['tfidf'], A['num'],A['category'])
        print(A['potential'], "Inppotential calculation")
        if (coll_content.isReceivedData()):
           coll_content.saveBuffer(A, 'Potential', root)
           return
        if root.children:
            for children in root.children:
        #        print("Category of children")
         #       print(children.category)
                if A['category'] == children.category:
                    flag = True
                    self.updatePrototype(A, children)
                    print(" Returned after protype updtaion",A['potential'])
                   # if (coll_content.isReceivedData()):
                    #    coll_content.saveBuffer(A, 'Update', children)
                     #   coll_content.isDataReceived()
                    self.comparePotential(A, children)
            if not flag:
                c_parent = root.add_child(A)
                self.insertPrototype(A, c_parent)
        else:
            c_parent = root.add_child(A)
            self.insertPrototype(A, c_parent)
        return
        '''for children in root.children:
            if A['category'] == children.category:
                self.insertPrototype(A, children)'''

    def calculatePotential(self, tfidf,k, category):
        A = {}
        A['category']= category
        A['tfidf'] = tfidf
        A['num'] = k
       # print("k value",k )

        if(k == 1):
            A['potential'] = 1
            return 1
        n = len(tfidf)
        a = self.valueB(tfidf, A['num'],A)
        b = A['num'] - 1
        c = self.norm(A['tfidf'])
        # print("Potential basic", a, b, c)
        potential = 1/(2-(a/(b*c)))
        A['potential'] = potential
        print("Potential ", potential, A['num'])
        return potential

    def norm(self, tfidf):
        return self.sqrt(self.calcsq(tfidf))

    def valueB(self, tfidf, k, A):
        b_v =self.calcdotProd(tfidf, self.value_b(tfidf, k, A))
        print('b value',b_v)
        return b_v

    def value_b(self, tfidf, k, A ):
        global prev_b_value
        print('prev b val',prev_b_value)
        b_value = []
        if(len(prev_b_value)==0):
            prev_b_value = [0]*len(tfidf)
        for i in range(0, len(tfidf)):
            try:
                a = self.local_sq_sum_ele(A, i)
                b = self.sqrt((tfidf[i]**2)/a)
                c = prev_b_value[i] + b
                b_value.append(c)
            except Exception as e:
                b_value.append(0)
        print('new b val',b_value)
        prev_b_value = b_value
        return b_value

    def local_sq_sum_ele(self, A, i):
        sum = 0
        for children in root.children:
            if children.category == A['category']:
                for prototype in children.children:
                    try:
                        sum = sum + (prototype.prototype['tfidf'][i])**2 #prototype length is 92
                    except Exception as e:
                        continue
                return sum

    def updatePrototype(self, A, children): # Category node
        for prototype in children.children:
           # print("protoype in beginning", prototype.prototype)
            try:
                if prototype.prototype == '':
                    for leaf_p in prototype.children:
                        up_potential = ((self.calcQ(A,prototype.children) - 1)*leaf_p.prototype['potential'])/((self.calcQ(A)-1)+((self.calcQ(A,prototype.children)-2)*(1/(leaf_p.prototype['potential'])-1))+(self.calcCosDist(A['tfidf'],leaf_p.prototype['tfidf'])))
                        print('up pot',up_potential)
                        leaf_p.setPotential(up_potential)
                else:
                    q_value = self.calcQ(A, children)
                    proto_potential = prototype.prototype['potential']
                    cos_distance = self.calcCosDist(A['tfidf'],prototype.prototype['tfidf'])
          #          print("Prototype Potential", prototype.prototype['potential'])
           #         print("Calculate Cosine Distance", self.calcCosDist(A['tfidf'],prototype.prototype['tfidf']))
                    up_potential = ((q_value - 1)*proto_potential)/((q_value-1)+((q_value-2)*(1/(proto_potential)-1))+(cos_distance))
                    print("Updated Potential", up_potential)
                    prototype.setPotential(up_potential)
            except:
             #   print("Exception in potential")
                prototype.setPotential(1)
        #print("Prototype updated")
        return

    def comparePotential(self, A, children):
        print("A potential", A['potential'])
        #print("comparing after updating")
        flag = True
        for prototype in children.children:
         #   print("Comparing potential", prototype.potential, A ['potential'])
            if prototype.prototype == '':
                for l_prototype in prototype.children:
                    if(l_prototype['potential']>A['potential']):
                        flag = False
            else:
                if(prototype.prototype['potential']>A['potential']):
                    flag = False
        if(flag):
          #  print("Inserted")
            if (coll_content.isReceivedData()):
                coll_content.saveBuffer(A, 'Compare-Insert', children)
                return
            self.insertPrototype(A, children)
        else:
            #print("Support Updated")
            if (coll_content.isReceivedData()):
                coll_content.saveBuffer(A, 'Compare-Update', children)
                return
            self.update_support(A, children)
        return

    def calcQ(self, A, children):
        support = 0
        for prototype in children.children:
           # print("Prototype Support Calculation",prototype.prototype['support'])
            if prototype.prototype == '':
                if(A['category'] == prototype.category):
                    support = support + prototype['support']
            else:
                support =support+ prototype.prototype['support']
        return support

    def update_support(self, A, children):
        min_dist = sys.maxsize
        temp_proto = {}
        coll_content = collectContent()
        for prototype in children.children:
            if prototype.prototype == '':
                for l_children in prototype.children:
                    print("L children", type(l_children))
                    dist_list = [x - y for x,y in zip(A['tfidf'], l_children['tfidf'])]
                    dist = self.norm(dist_list)
                    if dist<min_dist:
                        min_dist = dist
                        temp_proto = l_children
            else:
                dist_list = [x - y for x, y in zip(A['tfidf'], prototype.prototype['tfidf'])]
                dist = self.norm(dist_list)
                if dist < min_dist:
                    min_dist = dist
                    temp_proto = prototype.prototype
            temp_proto['support'] = temp_proto['support'] + 1
            if (coll_content.isReceivedData()):
                coll_content.saveBuffer(A, 'Compare-Insert', children)
                return
#            coll_content.isDataReceived()
        return

    def insertPrototype(self,A, parent):
        A['support'] = 1
        parent.addChildren(A)
        col_con = collectContent()
        #for children_p in parent.children:
         #   print("Check Insertion",  children_p.prototype['category'], parent.category)
          #  print("len",len(parent.children))
        # if (coll_content.isReceivedData()):
        #     col_con.saveBuffer(A, 'Compare-Insert', parent)
        #     return
        self.mergePrototype(parent)
        self.removePrototype_c(A, parent)
        return

    def printTree(self, root):
        print("no of categories", len(root.children))
        for children in root.children:
            print("Category of children", children.category)
            print("No of prototypes ", children.category, len(children.buffer))
            for p_children in children:
                if not p_children.A == '':
                    print("protoype", p_children.A['support'], p_children.A['category'],p_children.A['potential'], p_children.A['tfidf'])
                else:
                    print("After merging - Merged children")
                    for prototype in p_children:
                        print("prototype", prototype.A["support"], prototype.A["category"], prototype.A["potential"], prototype.A["tfidf"])
                    print("")
        return

    def calcCosDist(self, Ak ,Ap):
        #print("ksalhcksak")
        dist = 1 - ((self.calcdotProd(Ak, Ap))/(self.sqrt((self.calcsq(Ak))*(self.calcsq(Ap)))))
        #print("Cosine distance", dist)
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
        sum_d =0
        try:
            if(len(A)!= 0 and len(B)!=0):
                sum_d = sum([A[i]*B[i] for i in range(0, min(len(A), len(B)))])
                print('dottt vali', sum_d)
            else:
                sum_d = 0
        except Exception as e:
            print(e)
        return sum_d

    def mergePrototype(self, children):
        class_prototype = []
        merge_prototype = []
        c_tfidf =[]
        for prototype in children.children:
            if not prototype is '':
                class_prototype.append(prototype)
        #print("Length before merging", len(class_prototype))

        if(len(class_prototype)>2):
#            print(" Trying to merge")
            for i, prototype in enumerate(class_prototype):
                for j, prototype_c in enumerate(class_prototype):
                    if i< j and not prototype == prototype_c:

                        dist = self.calcCosDist(prototype.prototype['tfidf'], prototype_c.prototype['tfidf'])
                        #print("Distance", dist)
                        if dist < 0.97:
                            # change this threshold value
                            if prototype not in merge_prototype:
                                merge_prototype.append(prototype)
                            if prototype_c not in merge_prototype:
                                merge_prototype.append(prototype_c)
        if len(merge_prototype) > 0:
            P_node = TreePrototypeNode('', tokens=[] )
            children.children.append(P_node)
            P_node.setParent(children)
            tokens = []
            for prototype in merge_prototype:
 #               print(prototype.prototype)
                for tfidf in prototype.prototype['tfidf']:
                    tokens.append(tfidf)
                P_node.add_prototype(prototype.prototype)
                children.removeProto(prototype)
                P_node.setToken(tokens)

                # P_node.remove_children(prototype.prototype)

        return

    def removePrototype_c(self,A, children):

        for prototype in children.children:
            if prototype.prototype == '':
                for pchildren in prototype.children:
                    membership_value = self.calcMembership(A, pchildren)
                    if membership_value > math.exp(-1) and membership_value != 1:
                        self.removePrototype(children, prototype)
            else:
                membership_value = self.calcMembership(A, prototype)
                if membership_value > math.exp(-1) and membership_value != 1:
                    self.removePrototype(children, prototype)
  #          print("Calculate Membership", membership_value, math.exp(-1))
   #             print(" Prototype removes")
        self.combinePrototype(children)
        return

    def combinePrototype(self, children):
        coll_content = collectContent()
        class_prototype = []
        for prototype in children.children:
            if prototype.prototype is '':
                for children in prototype.children:
                    prototype.parent.addChildren(children)
                    prototype.remove_children(children)
                class_prototype.append(prototype)
#        print("Trying to reach")
#        coll_content.isDataReceived()
        return

    def updateTree(self):
        pass

    def find_min(self, tfidf, class_prototype):
        min_dist = sys.maxsize
        for i in range(0, len(class_prototype)):
            for j in range(i+1, len(class_prototype)):
                dist = self.calcCosDist(class_prototype[i], class_prototype[j])
                if dist < min_dist:
                    min_dist = dist

    def calcMembership(self, A, prototype):
        sigma_value = self.calcSigma(A,prototype)
 #       print("signma value", sigma_value)
        if(sigma_value != 0):
            try:
                a = self.calcCosDist(A['tfidf'], prototype.prototype['tfidf'])/sigma_value
            except:
                a = self.calcCosDist(A['tfidf'], prototype['tfidf'])/sigma_value
            check = math.exp((-0.5)*a)
            # print('mem',check,'cosin',self.calcCosDist(A['tfidf'], prototype['tfidf']),'sogm',sigma_value)
            return check
        else:
            return 0

    def calcSigma(self, A, prototype):
        global N
        print('prev sigma',self.prev_sigma_value)
        v1 = self.calcsq(self.prev_sigma_value)
        print('calq sig',v1)
        try:
            v2 = self.calcCosDist(A['tfidf'], prototype.prototype['tfidf'])
        except:
            v2 = self.calcCosDist(A['tfidf'], prototype['tfidf'])
        v3 = self.calcsq(self.prev_sigma_value)
        sig = self.sqrt(v1+(self.calcsq(v2)-v3)/N)
        self.prev_sigma_value = sig
        print('calculated dig',self.prev_sigma_value)
        return sig


    def calcCosDist(self, Ak ,Ap):

        dist = 1 - ((self.calcdotProd(Ak, Ap))/(self.sqrt((self.calcsq(Ak))*(self.calcsq(Ap)))))
        return dist
    #
    # def calcdotProd(self, A, B):
    #     if(len(A)!= 0 and len(B)!=0):
    #         print('in dot if')
    #         sum_d = sum([A[i]*B[i] for i in range(0, min(len(A), len(B)))])
    #     else:
    #         sum_d = 0
    #     print('dot value',sum_d)
    #     return sum_d

    def norm(self, tfidf):
        return self.sqrt(self.calcsq(tfidf))

    def generateTokens(self, mergeArray):
        token_list = []
        tmp_array= []
        token = ''
        for prototype in mergeArray:
            for i, token in enumerate(dictionary_vocab):
                if token in tmp_array:
                    pass
                elif prototype['tfidf'][i]> 0.5:
                    token_list.append(token)
                else:
                    if token in tmp_array:
                        tmp_array[token] = tmp_array[token] + prototype['tfidf'][i]
                    tmp_array.append({token: prototype['tfidf'][i]})
        return token


    def removePrototype(self, children, prototype):
  #      print("Removing prototype", children)
        children.removeProto(prototype)
        self.combinePrototype(children)
        return

    def searchPrototype(self):
        pass

class Preprocessing():
    def stop_words(self):
        pass
    def tokenize(self, text, category = ''):
        if(text == ''): return
        tokens = text.replace("\n", '').split(" ")
        tokens =list(filter(None, tokens))
        tokens = [re.sub('[^A-Za-z0-9]+', '', token) for token in tokens ]
        tokens = [token.lower() for token in tokens]
        return tokens

    # Stop word removal - Standaed nltk library
    def stopword_elimination(self, tokens,category = ''):
        tokens = [token for token in tokens if token not in stopword]
        return tokens

    #Snowball stemmer
    def stemming(self, tokens, category = ''):
        coll_content = collectContent()
        tokens = [stemmer.stemWord(token) for token in tokens]
        return tokens

    def storeDict(self, tokens, category = ''):
        tf_idf = C_tfidf()
        for token in tokens:
            if token in dictionary_vocab:
                dictionary_vocab[token] += 1
            else:
                dictionary_vocab[token] = 1
        return

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
        return v1

    def calctfidf(self, v1, category):
        global N
        tfidf = []
        index = 0
        v2 = np.asarray(dictionary_vocab.values())
        for i in dictionary_vocab:
            try:
                temp_v = N/dictionary_df[i]
                tfidf.append(v1[index]*math.log(1+temp_v, 10))
                index = index+1
            except:
                continue
        k = N
        return tfidf

class classifyPrototype():
    def classify(self, data):
        tfidf = C_tfidf()
        preprocess = Preprocessing()
        buildTree = BuildTree()
        buffer = []
        A = {}
        tokens = preprocess.tokenize(data['description'])
        tokens = preprocess.stopword_elimination(tokens)
        tokens = preprocess.stemming(tokens)
        preprocess.storeDict(tokens)
        preprocess.storeDicdf(tokens)
        v1 = tfidf.calctf(tokens,'')
        v1 = tfidf.calctfidf(v1,'')
        A['tfidf'] = v1
        tfidf = v1
        minVal = sys.maxsize
        category = ''
        global root
        for category in root.children:
            for children in category.children:
                if children.prototype  == '':
                    if buildTree.calcCosDist(children.tokens , tfidf) > 0.95:
                        for child in children.children:
                            try:
                                cosDist = buildTree.calcCosDist(child.prototype['tfidf'],tfidf)
                            except:
                                cosDist = buildTree.calcCosDist(child['tfidf'], tfidf)
                            if cosDist < minVal:
                                try:
                                    category = child.prototype['category']
                                except:
                                    category = child['category']
                                minVal = cosDist
                else:
                    cosDist = buildTree.calcCosDist(tfidf, children.prototype['tfidf'])
                    if cosDist < minVal:
                        category = children.prototype['category']
                        minVal = cosDist
        print("category")
        return category

class collectContent():


    def data(self, description, tag):
        global root
        if(description == ''): return
        A = {}
        #A['description'] = description
        A['category'] = tag
        build_tree = BuildTree()
        tfidf = C_tfidf()
        preprocess = Preprocessing()
        tokens = preprocess.tokenize(description, tag)
        tokens = preprocess.stopword_elimination(tokens, tag)
        tokens =preprocess.stemming(tokens, tag)
        preprocess.storeDict(tokens, tag)
        preprocess.storeDicdf(tokens)
        v1 = tfidf.calctf(tokens, tag)
        v1 = tfidf.calctfidf(v1, tag)
        A['tfidf'] = v1
        if(coll_content.isReceivedData()):
           self.saveBuffer(A, 'Preprocess', root)
           return
        build_tree.createNode(root, A)
        return

    def saveBuffer(self, A, status, node):
        curr_A = {}
        curr_A['prototype'] = A
        curr_A['status'] = status
        node.buffer.append(curr_A)
        return


    def dataReceived(self):

        classify = classifyPrototype()
        global data_buff
        global N
        data = {}
        total = 0
        positive = 0
        number = [1, 2, 3, 4]
        with open('inddata.csv', 'r+') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                num_generate = random.choice(number)
                if row[2] == 'Classify':
                    data['description'] = row[0]
                    data['category'] = row[1]
                    category = classify.classify(data)
                    print(category, data['category'])
                    if(category == data['category']):
                        positive = positive + 1
                    total = total + 1
                    print("Result", positive, total)
                elif num_generate == 1:
                    time.sleep(1)
                    data['description'] = row[0]
                    data['category'] = row[1]

                elif num_generate == 2:
                    time.sleep(2)
                    data['description'] = row[0]
                    data['category'] = row[1]

                elif num_generate == 3:
                    time.sleep(3)
                    data['description'] = row[0]
                    data['category'] = row[1]

                elif num_generate == 4:
                    time.sleep(1)
                    data['description'] = row[0]
                    data['category'] = row[1]

                lock.acquire()
                N = N + 1
                data_buff.append(data)
                # self.isDataReceived(data)
                lock.release()
        print("Result", positive, total)
        return

    def isDataReceived(self):
        global data_buff
        global N
        temp_data_buff = []
        clear_Buff = clearBuff()
        if self.isReceivedData():
 #           lock.acquire()
            temp_data_buff = copy.deepcopy(data_buff)
            data_buff.clear()
  #          lock.release()

        # if data:
        #     self.data(data['description'], data['category'])
        if len(temp_data_buff) == 0 and self.has_buffer_filled():
            clear_Buff.clearBuffers()

        elif len(temp_data_buff) == 0:
            while (len(data_buff) == 0): continue
            return
        elif len(temp_data_buff) == 1:
            self.data(temp_data_buff[0]['description'], temp_data_buff[0]['category'])
        else :
            for data_bu in temp_data_buff:
                initial_buffer.append(data_bu)
            coll_buff_data = initial_buffer[0]
            del initial_buffer[0]
            self.data(coll_buff_data['description'], coll_buff_data['category'])
        return

    def isReceivedData(self):
        global data_buff
        if len(data_buff)>0:
            return True
        return False

    def collectData(self):
        global initial_buffer
        global  data_buff
        data_ind = {}
        with atomic_write("inddata.csv", overwrite=True)as f:
            with open("newdata.csv") as f1:
                reader = csv.reader(f1, delimiter='\t')
                for line in reader:
                    if len(line)> 0:
                        data_ind['action'] = line[2]
                        data_ind['category'] = line[1]
                        data_ind['description'] = line[0]
                        data_buff.append(data_ind)
        f.truncate("newsdata.csv")
        os.remove("newsdata.csv")
        if len(data_buff) == 0  and self.has_buffer_filled():
            clearBuff.clearBuffers()
        elif len(data_buff) ==0:
            while(len(data_buff) == 0): continue
            self.collectData()
        elif len(data_buff) == 1:
            self.data(data_buff[0]['description'], data_buff[0]['category'])
        else:
            for data_bu in data_buff:
                initial_buffer.append(data_bu)
            coll_buff_data = initial_buffer[0]
            del initial_buffer[0]
            self.data(coll_buff_data['description'], coll_buff_data['category'])
        return

    def has_buffer_filled(self):
        global root
        global initial_buffer

        if not len(initial_buffer) == 0:
            return True
        else:
            if not len(root.buffer) == 0:
                return True
            for children in root.children:
                if not len(children.buffer) == 0:
                    return True
                for prototype in children.children:
                    if not len(prototype.buffer) == 0:
                        return True
        return False

class clearBuff():

    def clearBuffers(self):
     #   print("in clear buffer")
        global root
        global initial_buffer
        build_tree= BuildTree()
        collectCont = collectContent()
        if len(initial_buffer)>0:
            coll_buff_data = initial_buffer[0]
            del initial_buffer[0]
            collectCont.data(coll_buff_data['description'], coll_buff_data['category'])
        else:
            if not len(root.buffer) == 0:
                coll_buffer_data = root.buffer[0]
                del root.buffer[0]
                build_tree.createNode(root, coll_buffer_data['prototype']);

            for cat_children in root.children:
                if not len(cat_children.buffer) == 0:
                    coll_buff_data = cat_children.buffer[0]
                    del cat_children.buffer[0]
                    if(coll_buff_data['status'] == 'Update'):
                        build_tree.comparePotential(coll_buff_data['prototype'], cat_children)
                    if (coll_buff_data['status'] == 'Compare-Insert'):
                        build_tree.insertPrototype(coll_buff_data['prototype'], cat_children)
                    if (coll_buff_data['status'] == 'Compare-Update'):
                        build_tree.update_support(coll_buff_data['prototype'], cat_children)
                for pro_children in cat_children.children:
                    if pro_children.prototype == "":
                        # merge prototype and update
                        # remove prototype and
                        pass
        return


if __name__ == '__main__':
    coll_content = collectContent()
    p1 = Thread(target=coll_content.dataReceived, args=())
    p1.start()

    while True:
        p2 = Thread(target=coll_content.isDataReceived, args=())
        p2.start()
        p2.join()

