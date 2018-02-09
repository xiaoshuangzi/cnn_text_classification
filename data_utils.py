#!/usr/bin/python
# -*- coding:utf-8 -*-
import nltk
import os
import re
import string
import pickle
import codecs
import time
import json
import torch
from nltk import word_tokenize
import torchwordemb
import numpy as np
import csv
import jieba
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

def slit_to_words(source_path, splited_path, mode):
	csvFile = open(source_path, 'r')
	reader = csv.reader(csvFile)
	writer = open(splited_path, 'w')
	data = {}
	for row in reader:
		user_id = row[0].encode("utf-8")
		comment_text = row[1].encode("utf-8")
		# if (comment_text.startswith("Discuss")):
		# 	continue
		origin_words = jieba.cut(comment_text)
		# no_stop_words = [word for word in origin_words if word not in stop_word]
		splited_text = " ".join(origin_words)
		if mode == "train":
			label = row[2]
		else:
			label = str(1)
		writer.write("\t".join([splited_text, label, str(user_id)]) + "\n")

def get_tf_idf_info(splited_path):
	vocab = {}
	word_appear_text_count = {}
	word_count_all_comment = []
	word_num = 0
	text_num = 0
	with open(splited_path, 'r') as f_r:
		for line in f_r:
			text_num += 1
			segs = line.strip().split("\t")
			comment_text = segs[0]
			label = segs[1]
			words = comment_text.split(" ")
			word_num_each_comment = len(words)
			word_count_each_comment = {}
			for word in words:
				word_num += 1
				# if word not in word_count_each_comment:
				# 	word_count_each_comment[word] = 1
				# else:
				# 	word_count_each_comment[word] += 1
				if word not in vocab:
					vocab[word] = 1
				else:
					vocab[word] += 1
				# word_count_all_comment.append(word_count_each_comment)
			# for word in word_count_each_comment:
			# 	if word not in word_appear_text_count:
			# 		word_appear_text_count[word] = 1
			# 	else:
			# 		word_appear_text_count[word] += 1

	print ("vocab size is: " + str(len(vocab)))
	vocab = sorted(vocab.items(), key = lambda item : item[1], reverse = True)
	# print "word num is: " + str(word_num)
	# print "text num is: " + str(text_num)

	# json.dump(word_appear_text_count, open("./data/word_appear_text_count", 'w'))
	# json.dump(word_count_all_comment, open("./data/word_count_all_comment", 'w'))
	for i in range(20):
		print (vocab[i][0].encode('utf-8'))
	return vocab, word_num, text_num, word_appear_text_count, word_count_all_comment

def get_tf_idf_vector(vocab, word_num, text_num, word_qppear_text_count, word_count_all_comment, splited_path):
	vector = []
	count = 0
	with open(splited_path, 'r') as f_r:
		for line in f_r:
			segs = line.strip().split("\t")
			comment_text = segs[0]
			label = segs[1]
			words = comment_text.split(" ")
			vector.append([0 for i in range(len(vocab))])
			for i in range(len(vocab)):
				if vocab[i] in word_count_all_comment[count]:
					tf = word_count_all_comment[count][vocab[i]]/len(words)
					include_text_num = 1 if vocab[i] not in word_appear_text_count else word_appear_text_count[vocab[i]]
					idf = log(text_num/include_text_num)
					vector[count][i] = tf * idf
			count += 1
	print (vector[0])
	return vector

def compute_tfidf(splited_data_path):
	corpus = []
	count = 0
	with open(splited_data_path, 'r') as f:
		for line in f:
			count+=1
			if (count > 100):
				break
			corpus.append(line.strip().split("\t")[0])
	vectorizer = CountVectorizer()
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
	word = vectorizer.get_feature_names()
	weight = tfidf.toarray()
	for i in range(len(weight)):
		print (u"--------第",i,u"个文本的词语权重")
		for j in range(len(word)):
			print( word[j], weight[i][j] )
if __name__ == '__main__':
	data_path = "./data/"
	train_data_path = data_path + "train_first.csv"
	splited_data_path = data_path + "split_train_data.txt"
	stop_word_path = data_path + "stop_words_list"
	slit_to_words(train_data_path, splited_data_path, stop_word_path)
	# vocab, word_num, text_num, word_appear_text_count, word_count_all_comment = get_tf_idf_info(splited_data_path)
	# vector = get_tf_idf_vector(vocab, word_num, text_num, word_appear_text_count, word_count_all_comment, splited_data_path)
	# get_tf_idf_info(splited_data_path)