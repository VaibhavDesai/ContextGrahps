from pymongo import *
import tika
import sys
from tika import parser
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup

import urllib2
import json
import re, string

from py2neo import authenticate, Graph, Node,Relationship,Path, Rev, rel
from nltk.corpus import wordnet
import json
from collections import deque
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

class Extractor:

    def __init__(self):
        authenticate("localhost:7474", "neo4j", "vai")
        global graph
        graph = Graph("http://localhost:7474/db/data/")
        global minSimilarityScore
        minSimilarityScore=0.2

    def extactWordList(self,path):
        #parsed = parser.from_file(path)
        #fileContent = parsed['content']
        file = open(path)
        fileContent = file.read().replace('\n', ' ')
        print fileContent
        fileContent=fileContent.decode('utf-8','ignore').encode("utf-8")
        fileContent = re.sub(r"\n", " ", fileContent)
        fileContent = re.sub(r"\t", " ", fileContent)
        self.createKeyValue(fileContent)


    def contextBuilder(self,content):

        graphDic = []
        contentWordList = content.split()
        print contentWordList
        filteredWordsList = self.removeStopWords(contentWordList)
        #stemmedWordList = self.stemmingWordList(filteredWordsList)
        filteredWordsList = self.removeSynonyms(filteredWordsList)
        similarityDic = self.similarityScore(filteredWordsList)

        i=0.1
        minSimilarityScores = []
        minSimilarityScoreGraphs = []
        minSimilarityGraphScores = []
        while i <= 1.0:
            minSimilarityScores.append(i);
            score , graphs = self.disjointGraph(similarityDic,i)
            minSimilarityScoreGraphs.append(graphs)
            minSimilarityGraphScores.append(score)
            i = i + 0.1

        peak_value = max(minSimilarityGraphScores)
        optimalScore = minSimilarityScores[minSimilarityGraphScores.index(peak_value)]
        optimalGraphs = minSimilarityScoreGraphs[minSimilarityGraphScores.index(peak_value)]

        return optimalGraphs,optimalScore

    def createKeyValue(self,text):
        pattern = re.compile(r'<key>(.*?)</key>:<property>(.*?)</property>')
        matches = pattern.findall(text)
        result = []
        print matches

        for element in matches:
            keyValueGraph = {}
            keyValueGraph['key'] = element[0]
            keyValueGraph['contextGraph'],keyValueGraph['graphScore'] = self.contextBuilder(element[1]) #value.
            result.append(keyValueGraph)

        print result


    def removeDuplicates(self,wordList):
    	wordSet =  set(wordList)
        return list(wordSet)

    def removeStopWords(self,wordList):
    	newList = []
    	for word in wordList:
            word = word.lower()
            if word not in stopwords.words('english') and not word.startswith('http://'):
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
                key = wordList[i]+'-'+wordList[j]
                if len(wordnet.synsets(wordList[i]))> 0  and len(wordnet.synsets(wordList[j]))> 0:
                    similarityJson[key] = wordnet.wup_similarity(wordnet.synsets(wordList[i])[0],wordnet.synsets(wordList[j])[0])
                j = j+1
            i = i+1

        return similarityJson

    def disjointGraph(self,similarityJson,minSimilarityScore):
        keyWordList = []
        minSimilarityScoreGraph = []
        minSimilarityScoreDic = {}
        tempSimilarityJson = {key: value for key, value in similarityJson.items() if value > minSimilarityScore}
        if len(tempSimilarityJson) == 0:
            print "Graph Score with minSimilarityScore:",minSimilarityScore," is:0"
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

        print "Graph Score with minSimilarityScore:",minSimilarityScore," is:",(graphScore/numberOfDisjointGraphs)
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

        print otherWordsList

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

    def graphConstruct(self,similarityJson):

        graph.delete_all()
        #for key, value in similarityJson.iteritems():
        #    if(value > minSimilarityScore):
                #print key ,'-',value

        for key, value in similarityJson.iteritems():
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
    print "Enter file path and wordListCount\n"
else:
    print extractor.extactWordList(sys.argv[1])
