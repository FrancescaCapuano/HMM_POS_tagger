from pathlib import Path

import urllib.request

import zipfile

import os

import nltk
from nltk.corpus.reader.conll import * 
from nltk.probability import *
from nltk import word_tokenize
from nltk.util import ngrams

import numpy




class Conll:


	'''
	instantiates a Conll object.
	The file has to be a Conll file with two columns: 
	the first one containing the words of the corpus, 
	the second one containing the part-of-speech tags.
	'''
	def __init__(self,filename):
		self.text=nltk.corpus.reader.conll.ConllCorpusReader(os.path.dirname(os.path.abspath(__file__)),fileids=filename,columntypes=['words','pos'])
		self.tagged_sents=self.text.tagged_sents()
		self.tagged_words=self.text.tagged_words()
		self.tags=tuple(set(tag for (word,tag) in self.tagged_words))
		self.vocabulary=tuple(set(word for (word,tag) in self.tagged_words))




	#returns an MLEProbDist object of the initial tags.
	def pdist_initial_tags(self):
		initial_tags=[]
		for sent in self.tagged_sents:
			initial_tags.append(sent[0][1]) #appends first tag of each sentence to initial_tags
		fdist_initial_tags=FreqDist(initial_tags)
		return MLEProbDist(fdist_initial_tags)




	'''
	returns a ConditionalProbDist object of the tags bigrams.
	cpdist[tag1].prob(tag2) returns the probability of occurrence of tag2, given tag1.
	'''
	def cpdist_tags_bigrams(self):
		tags_bigrams=[]
		for sent in self.tagged_sents:
			sent_tags=[]
			for tupl in sent:
				sent_tags.append(tupl[1]) #the tags of a single sentence
			bigrams=list(nltk.bigrams(sent_tags))
			tags_bigrams+=bigrams
		cfdist_tags = ConditionalFreqDist(tags_bigrams)
		return nltk.ConditionalProbDist(cfdist_tags, nltk.MLEProbDist)




	'''
	returns a ConditionalProbDist object of the words, given the tags .
	cpdist[tag].prob(word) returns the probability of occurrence of word, given tag.
	'''
	def cpdist_tags_words(self,distribution):
		cfdist_emissions = ConditionalFreqDist([(tag,word) for (word,tag) in self.tagged_words])
		if distribution=='Laplace':
			return ConditionalProbDist(cfdist_emissions, nltk.LaplaceProbDist,bins=len(self.vocabulary))
		elif distribution=='MLE':
			return ConditionalProbDist(cfdist_emissions, nltk.MLEProbDist)