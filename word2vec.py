from __future__ import division

import collections
import math
import os
import random
import itertools
from Word2vecDataSet import Word2vecDataSet

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

INPUT_FILE_NAME = 'honghua_mars_p13n_1d_new_cate_query_impression_20161204_processed'
OUTPUT_FILE_NAME = 'honghua_mars_p13n_1d_new_cate_query_impression_20161204_word2vec'
INPUT_FILE_PATH = '../data/' + INPUT_FILE_NAME
OUTPUT_FILE_PATH = '../data/' + OUTPUT_FILE_NAME

# parameter for word2vec
BATCH_SIZE = 128
EMDEDDING_SIZE = 128
SKIP_WINDOW = 1
NUM_SKIP = 2
NUM_NEGATIVE_SAMPLED = 64

# parameters for learning
LEARNING_RATE = 1.0e-3
NUM_EPOCH = 5

def main():
	dataSet = Word2vecDataSet()
	dataSet.loadDataset(INPUT_FILE_PATH, 10)
	vocabularySize = dataSet.get_vocabulary_size()
	totalWordNumber = dataSet.get_total_word_number()

	# def network
	graph = tf.Graph()

	with graph.as_default():

		# input data
		trainInput = tf.placeholder(tf.int32, shape = [BATCH_SIZE])
		trainLabel = tf.placeholder(tf.int32, shape = [BATCH_SIZE ,1])

		embeddings = tf.Variable(
			tf.random_uniform([vocabularySize, EMDEDDING_SIZE], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, trainInput)

		# construct the variables for the NCE loss
		nceWeights = tf.Variable(
			tf.truncated_normal([vocabularySize, EMDEDDING_SIZE], stddev = 1.0 / math.sqrt(vocabularySize)))
		nceBiases = tf.Variable(tf.zeros([vocabularySize]))

		# Compute the average NCE loss for the batch.
		# tf.nce_loss automatically draws a new sample of the negative labels each
		# time we evaluate the loss.

		loss = tf.reduce_mean(
			tf.nn.nce_loss(weights = nceWeights,
				biases = nceBiases,
				labels = trainLabel,
				inputs = embed,
				num_sampled = NUM_NEGATIVE_SAMPLED,
				num_classes = vocabularySize))

		optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

		# Compute the cosine similarity between minibatch examples and all embeddings
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
		normalizedEmbeddings = embeddings / norm

		# Add variable initializer.
		init = tf.initialize_all_variables()

	with tf.Session(graph=graph) as session:
		init.run()
		print 'initialize the network'

		avergeLoss = 0
		stepNumber = totalWordNumber * NUM_SKIP // BATCH_SIZE
		for epoch in xrange(NUM_EPOCH):
			for step in xrange(stepNumber):
				batch, label = dataSet.get_batch(BATCH_SIZE, NUM_SKIP, SKIP_WINDOW)
				_, lossVal = session.run([optimizer, loss], feed_dict = {
																		trainInput: batch,
																		trainLabel: label})
				avergeLoss += lossVal

				if step % 2000 == 0:
					if step > 0:
						avergeLoss /= 2000
					print 'Average loss at (%d, %d): %f'%(epoch, step, avergeLoss)
					avergeLoss = 0

		# for step in xrange(1):
		# 	batch, label = dataSet.get_batch(BATCH_SIZE, NUM_SKIP, SKIP_WINDOW)
		# 	_, lossVal = session.run([optimizer, loss], feed_dict = {
		# 															trainInput: batch,
		# 															trainLabel: label})
		# 	avergeLoss += lossVal

		# 	if step % 2000 == 0:
		# 		if step > 0:
		# 			avergeLoss /= 2000
		# 		print 'Average loss at (%d, %d): %f'%(0, step, avergeLoss)
		# 		avergeLoss = 0

		wordVector = normalizedEmbeddings.eval()

	# save result to file:
	wordDictionary = dataSet.get_word_dictionary()
	with open(OUTPUT_FILE_PATH, 'w') as outFile:
		print 'Save word and vector to file: %s'%(OUTPUT_FILE_PATH)
		for word, idx in wordDictionary.items():
			str = word + ' ' + ' '.join('%f'%(data) for data in wordVector[idx]) + '\n'
			outFile.write(str)
		print 'save done!'

if __name__ == '__main__':
	main()
