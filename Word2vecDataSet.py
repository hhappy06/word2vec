from __future__ import division

import collections
import math
import os
import random
import itertools

import numpy as np
import matplotlib.pyplot as plt

class Word2vecDataSet():
	"""docstring for Word2vecDataset"""
	def __init__(self):
		# self.trainingSet = []
		# self.testingSet = []
		# self.validationSet = []
		self.size = 0
		self.currentCorpusPos = 0
		self.adsetPos = 0

		self.totalWordNumber = 0
		self.vocabularySize = 0
		self.corpusIdxList = []
		self.wordCount = []
		self.wordDictionary = {}
		self.wordReverseDictionary = {}

	def loadDataset(self, fileName, leastFrequence = 10):
		corpus = self._read_data(fileName)
		self.vocabularySize, \
		self.corpusIdxList, \
		self.wordCount, \
		self.wordDictionary, \
		self.wordReverseDictionary = self._build_corpus_dictionary(corpus, leastFrequence)
		del corpus
		self.size = len(self.corpusIdxList)
		self.currentCorpusPos = 0
		self.adsetPos = 0

		print ('corpus length: %d'%(len(self.corpusIdxList)))
		print ('total word: %d'%(self.totalWordNumber))
		print ('vocabulary size: %d'%(self.vocabularySize))

	def get_vocabulary_size(self):
		return self.vocabularySize

	def get_total_word_number(self):
		return self.totalWordNumber

	def get_word_dictionary(self):
		return self.wordDictionary

	def get_word_reverse_dictionary(self):
		return self.wordReverseDictionary

	def get_batch(self, batchSize, numSkips, skipWindow):
		assert numSkips <= 2 * skipWindow
		batch = np.ndarray(shape = (batchSize), dtype = np.int32)
		label = np.ndarray(shape = (batchSize, 1), dtype = np.int32)
		span = 2 * skipWindow + 1

		currentBatchSize = 0
		while currentBatchSize < batchSize:
			buffer = []
			currentCorpus = self.corpusIdxList[self.currentCorpusPos]
			for pos in np.arange(self.adsetPos - skipWindow, self.adsetPos + skipWindow, 1):
				if pos >= 0 and pos != self.adsetPos and pos < len(currentCorpus):
					buffer.append(currentCorpus[pos])

			sampledNumber = min(batchSize - currentBatchSize, len(buffer))
			sampledData = random.sample(buffer, sampledNumber)
			for data in sampledData:
				batch[currentBatchSize] = currentCorpus[self.adsetPos]
				label[currentBatchSize] = data
				currentBatchSize += 1

			if self.adsetPos < len(currentCorpus) - 1:
				self.adsetPos += 1
			else:
				self.adsetPos = 0
				self.currentCorpusPos = (self.currentCorpusPos + 1) % len(self.corpusIdxList)
				if self.currentCorpusPos % len(self.corpusIdxList) == 0:
					random.shuffle(self.corpusIdxList)

		return batch, label

	def _read_data(self, fileName):
		wordList = []
		if not os.path.exists(fileName):
			print 'file does not exists: %s'%(fileName)
			return wordList

		with open(fileName, 'r') as data:
			for item in data:
				wordList.append(item.rstrip('\n').split(' '))

		return wordList

	def _build_corpus_dictionary(self, corpus_data, least_frequence = 10):
		# flatten word list
		wordList = list(itertools.chain(*corpus_data))
		wordCollection = collections.Counter(wordList)
		frequentWord = wordCollection.most_common()

		# find least frequence word
		vocabularySize = len(frequentWord)
		for idx in xrange(len(frequentWord)):
			if frequentWord[-(idx+1)][1] >= least_frequence:
				vocabularySize = len(frequentWord) - idx + 1
				break

		wordCount = [['UNK',-1]]
		wordCount.extend(wordCollection.most_common(vocabularySize - 1))
		wordDictionary = {}
		for word, _ in wordCount:
			wordDictionary[word] = len(wordDictionary)
		
		# show word frequence 
		# plt.figure()
		# plt.plot([item[1] for item in frequentWord[7000:9000]])
		# plt.show()
		# exit()

		corpusIdxList = []
		unkCount = 0
		for item in corpus_data:
			tempList = []
			if len(item) == 1:
				continue
			for word in item:
				if word in wordDictionary:
					idx = wordDictionary[word]
				else:
					idx = 0
					unkCount += 1
				tempList.append(idx)
			corpusIdxList.append(tempList)
			self.totalWordNumber += len(tempList)
		wordCount[0][1] = unkCount
		wordReverseDictionary = dict(zip(wordDictionary.values(), wordDictionary.keys()))

		return vocabularySize, corpusIdxList, wordCount, wordDictionary, wordReverseDictionary
