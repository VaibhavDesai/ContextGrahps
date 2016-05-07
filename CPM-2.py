from pymongo import *
import tika
import sys
from tika import parser
from collections import Counter
import time
import os

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup

import urllib2
import json
import re, string
import operator

from py2neo import authenticate, Graph, Node,Relationship,Path, Rev, rel
from nltk.corpus import wordnet
import json
from collections import deque
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pyexcel

class Extractor:

    def __init__(self):
        authenticate("localhost:7474", "neo4j", "vai")
        global graph
        graph = Graph("http://localhost:7474/db/data/")
        global minSimilarityScore
        minSimilarityScore=0.2
        global fileName

    def extactWordList(self,path):
        #parsed = parser.from_file(path)
        #fileContent = parsed['content']
        fileName = os.path.basename(path)
        file = open(path)

        fileContent = file.read().replace('\n', ' ')
        print "The words in file are :",len(re.findall(r'\w+', fileContent))
        fileContent = fileContent.decode('utf-8','ignore')
        fileContent = re.sub(r"\n", " ", fileContent)
        fileContent = re.sub(r"\t", " ", fileContent)
        #self.createKeyValue(fileContent)
        self.contextBuilder(fileContent)

    def preprocessing(self,content):
        contentWordList = content.split()
        filteredWordsList = self.removeStopWords(contentWordList)
        #filteredWordsList = self.stemmingWordList(filteredWordsList)
        filteredWordsList = self.removeSynonyms(filteredWordsList)
        return filteredWordsList

    def removeDuplicates(self,wordList):
    	wordSet =  set(wordList)
        return list(wordSet)

    def removeStopWords(self,wordList):
    	newList = []
    	for word in wordList:
            word = word.lower()
            if word not in stopwords.words('english') and not word.startswith('http://') and not word.isdigit():
    			if re.match('[_+\-.,!@#$%^&*()?:;\/|<>]|(([a-zA-Z]+?)[_+\-.,!@#$%^?&*():;\/|<>])(|\s|$)',word):
	    			punctuations = '''!()-[]{};?:'"\,<>./?@#$%^&*_~'''
	    			no_punct=""
	    			for char in word:
	   					if char not in punctuations:
	   						no_punct = no_punct + char

	    			word = no_punct
	    		newList.append(word)

    	return newList

    def stemmingWordList(self,wordList):
    	stemmer = SnowballStemmer("english")
    	return [stemmer.stem(word) for word in wordList]

    def removeSynonyms(self,wordList):
        tempWordList = wordList;
        for word in wordList:
            if len(wordnet.synsets(word))> 0:
                synonyms = wordnet.synsets(word)[0].lemma_names()
                for item in wordList:
                    if item in synonyms and item != word:
                        tempWordList.remove(item)

        return tempWordList

    def similarityScore(self,wordList):
        similarityJson = {}
        i=0
        while (i<len(wordList)):
            j = i+1
            while(j<len(wordList)):
                if wordList[i] == wordList[j]:
                    break
                key = wordList[i]+'-'+wordList[j]
                if len(wordnet.synsets(wordList[i]))> 0  and len(wordnet.synsets(wordList[j]))> 0:
                    similarityJson[key] = wordnet.wup_similarity(wordnet.synsets(wordList[i])[0],wordnet.synsets(wordList[j])[0])
                j = j+1
            i = i+1

        return similarityJson

    def contextBuilder(self,content):

        graphDic = []
        tempSimilarityDic = {}
        filteredWordsList = self.preprocessing(content)
        temp1SimilarityDic = self.similarityScore(filteredWordsList)
        tempSimilarityDic.update((k, v) for k, v in temp1SimilarityDic.iteritems() if v is not None)

        #Method 1- Find recursively the first value where the graph breaks
        self.pruneGraph(tempSimilarityDic)


        #Method 2-Here the pruning is done incrementing threshold value by 1
        '''
        i=0.1
        minSimilarityScores = []
        minSimilarityScoreGraphs = []
        minSimilarityGraphScores = []
        while i <= 1.0:
            minSimilarityScores.append(i);
            score , graphs = self.disjointGraph(tempSimilarityDic,i)
            minSimilarityScoreGraphs.append(graphs)
            minSimilarityGraphScores.append(score)
            i = i + 0.1

        peak_value = max(minSimilarityGraphScores)
        optimalScore = minSimilarityScores[minSimilarityGraphScores.index(peak_value)]
        optimalGraphs = minSimilarityScoreGraphs[minSimilarityGraphScores.index(peak_value)]

        similarityDic = {}

        for key,value in tempSimilarityDic:
            if value is not None:
                similarityDic[key] = value

        print "count is :",len(similarityDic)
        return optimalGraphs,optimalScore,similarityDic
        '''

    def pruneGraph(self,tempSimilarityDic):
        sortedTempSimilarityTuple = sorted(tempSimilarityDic.items(), key=operator.itemgetter(1))

        for index in range (0,len(sortedTempSimilarityTuple)):
            sortedTempSimilarityTuple[index] = (sortedTempSimilarityTuple[index][0],format(sortedTempSimilarityTuple[index][1],'.2f'))

        self.CVSOutput(sortedTempSimilarityTuple)

        masterGraph = []
        subGraph = []
        tempValue = sortedTempSimilarityTuple[0][1]


        '''for tuple in sortedTempSimilarityTuple:
            if tuple[1] <= tempValue:
                subGraph.append(tuple)
            else:
                #print the subGraph here.
                for item in sortedTempSimilarityTuple:
                    if item not in subGraph:
                        print item
                        tempValue = tuple[1]
                        subGraph.append(tuple)


        #print masterGraph
        '''

    def CVSOutput(self,sortedTempSimilarityTuple):
        data = []
        for item in sortedTempSimilarityTuple:
            dataTuple = []
            relations = item[0].split("-")
            dataTuple.append(relations[0])
            dataTuple.append(relations[1])
            dataTuple.append(item[1])
            data.append(dataTuple)
        pyexcel.save_as(array = data, dest_file_name = 'testCSV.csv')

    def createKeyValue(self,text):
        pattern = re.compile(r'<key>(.*?)</key>:<property>(.*?)</property>')
        matches = pattern.findall(text)
        result = []
        for element in matches:
            keyValueGraph = {}
            keyValueGraph['key'] = element[0]
            keyValueGraph['contextGraph'],keyValueGraph['graphScore'],keyValueGraph['similarityDic'] = self.contextBuilder(element[1]) #value.
            result.append(keyValueGraph)

        for element in result:
            #self.graphConstruct(element)
            print element

    def disjointGraph(self,similarityJson,minSimilarityScore):
        keyWordList = []
        minSimilarityScoreGraph = []
        minSimilarityScoreDic = {}
        tempSimilarityJson = {key: value for key, value in similarityJson.items() if value > minSimilarityScore}
        if len(tempSimilarityJson) == 0:
            #print "Graph Score with minSimilarityScore:",minSimilarityScore," is:0"
            return 0,[]
        for key,value in tempSimilarityJson.iteritems():
            if(value > minSimilarityScore):
                words = key.split("-")
                keyWordList.append(words[0]) if words[0] not in keyWordList else None
                keyWordList.append(words[1]) if words[1] not in keyWordList else None

        keyWordList.sort()
        numberOfDisjointGraphs = 0
        graphScore = 0
        while True:
            if len(keyWordList):
                word = keyWordList[0]
                tempSimilarityJson,otherWordsList,disjointGraphScore = self.connectedWords(tempSimilarityJson,word)
                minSimilarityScoreGraph.append(otherWordsList)
                keyWordList = [item for item in keyWordList if item not in otherWordsList]
                graphScore = graphScore + disjointGraphScore
                numberOfDisjointGraphs += 1
            else:
                break

        #print "Graph Score with minSimilarityScore:",minSimilarityScore," is:",(graphScore/numberOfDisjointGraphs)
        return (graphScore/numberOfDisjointGraphs),minSimilarityScoreGraph

    def connectedWords(self,tempSimilarityJson,word):
        otherWordsList = [word]
        pairList = []
        recentPairList = []
        queue = deque([])
        graphScore = []
        while True:
            recentPairList = []
            for wordPair,score in tempSimilarityJson.iteritems():
                if word in wordPair:
                    graphScore.append(score)
                    recentPairList.append(wordPair)
                    pairList.append(wordPair)
                    otherWord = re.sub("-"+word+"|"+word+"-","",wordPair)
                    otherWordsList.append(otherWord) if otherWord not in otherWordsList else None
                    queue.append(otherWord) if otherWord not in queue else None

            for pair in recentPairList:
                if pair in tempSimilarityJson:
                    del tempSimilarityJson[pair]

            if len(queue) == 0:
                break

            word = queue.popleft()

        i=0
        possibleRelations = 0
        while i < len(otherWordsList):
            possibleRelations = possibleRelations+i
            i=i+1

        if len(otherWordsList) == 1:
            densityScore = 0
        else:
            densityScore = (len(pairList)*1.0)/possibleRelations

        #print "densityScore:",densityScore
        #print "mean:",self.mean(graphScore)
        #print "densityScore:",densityScore
        disjointGraphScore = self.mean(graphScore)*densityScore
        return tempSimilarityJson,otherWordsList,disjointGraphScore

    def mean(self,l):
        if len(l) == 1:
            return l[0]
        if len(l) == 0:
            return 0
        else:
            return reduce(lambda x, y: x + y, l) / float(len(l))

    def stddev(self,lst):
        """returns the standard deviation of lst"""
        variance = 0
        mn = self.mean(lst)
        for e in lst:
            variance += (e-mn)**2
        variance /= len(lst)

        return sqrt(variance)

    def graphConstruct(self,item):

        similarityDic = item['similarityDic']
        minSimilarityScore = item['graphScore']
        title = item['key']
        #source = fileName
        contextGraph = item['contextGraph']
        similarityJson = {}
        score =0
        for disjointGraph in contextGraph:
            for i in range(0,len(disjointGraph)-1):
                for j in range(i+1,len(disjointGraph)):
                    if similarityDic.get(disjointGraph[i]+"-"+disjointGraph[j], 0) > 0:
                        score = similarityDic.get(disjointGraph[i]+"-"+disjointGraph[j], 0)
                        similarityJson[disjointGraph[i]+"-"+disjointGraph[j]] = score
                    elif similarityDic.get((disjointGraph[j]+"-"+disjointGraph[i]), 0)>0:
                        score = similarityDic.get((disjointGraph[j]+"-"+disjointGraph[i]), 0)
                        similarityJson[disjointGraph[j]+"-"+disjointGraph[i]] = score

        print title, "----------------------"

        '''for key, value in similarityJson.iteritems():
            if(value > minSimilarityScore):
                words = key.split("-")
                node1 = graph.find_one("testing",property_key='name',property_value=words[0])
                node2 = graph.find_one("testing",property_key='name',property_value=words[1])
                if not node1 :
                    node1 = Node("testing",name=words[0])
                    graph.create(node1)
                if not node2 :
                    node2 = Node("testing",name=words[1])
                    graph.create(node2)

                relVal  = 'Score'
                if(value>7):
                    color = 'red'
                else:
                    color = 'green'
                graph.create(rel(node1,value,node2,{'color':color,'score':value}));
        '''

    def constructPlot(self,yAxis,xAxis):

        y_mean = [np.mean(yAxis) for i in xAxis]
        y_std_lower = [(self.mean(yAxis)-self.stddev(yAxis)) for i in xAxis]
        y_std_upper = [(self.mean(yAxis)+self.stddev(yAxis)) for i in xAxis]
        fig,ax = plt.subplots()
        data_line = ax.plot(xAxis,yAxis, label='Data', marker='o')
        mean_line = ax.plot(xAxis,y_mean, label='Mean', linestyle='--')
        std_line_lower = ax.plot(xAxis,y_std_lower,label='Standard Deviation',linestyle='--',color='red')
        std_line_upper = ax.plot(xAxis,y_std_upper,label='Standard Deviation',linestyle='--',color='red')

        # Make a legend
        legend = ax.legend(loc='upper right')
        plt.show()


extractor = Extractor()
if(len(sys.argv)<2):
    print "Enter file path\n"
else:
    start = time.clock()
    extractor.extactWordList(sys.argv[1])
    print "Time taken is:",(time.clock() - start)
