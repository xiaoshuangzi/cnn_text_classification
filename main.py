#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import model
import json
import re
import train
from dataset import comment_dset
from torch.utils.data import DataLoader
from data_utils import slit_to_words

parser = argparse.ArgumentParser(description = 'CNN text classificer')
# learning
parser.add_argument('-lr', type = float, default = 0.001)
parser.add_argument('-epochs', type = int, default = 256)
parser.add_argument('-batch-size', type = int, default = 64)
parser.add_argument('-log-interval', type = int, default = 1)
parser.add_argument('-test-interval', type = int, default = 100)
parser.add_argument('-save-interval', type = int, default = 500)
parser.add_argument('-save-dir', type = str, default = 'snapshot')
parser.add_argument('-early-stop', type = int, default = 1000)
parser.add_argument('-save-best', type = bool, default = True)

# data
parser.add_argument('-shuffle', action = 'store_true', default = False)
parser.add_argument('-data-dir', type = str, default = 'data/')
# model
parser.add_argument('-dropout', type = float, default = 0.5)
parser.add_argument('-max-norm', type = float, default = 3.0)
parser.add_argument('-embed-dim', type = int, default = 128)
parser.add_argument('-kernel-num', type = int, default = 100)
parser.add_argument('-kernel-sizes', type = str, default = '3,4,5')
parser.add_argument('-static', action = 'store_true', default = False)

# device
parser.add_argument('-device', type = int, default = -1)
parser.add_argument('-no-cuda', action = 'store_true', default = False)

# option
parser.add_argument('-snapshot', type = str, default = None)
parser.add_argument('-predict', type = str, default = None)
parser.add_argument('-test', action = 'store_true', default = False)

args = parser.parse_args()


def collate_fn(batch):
		batch.sort(key = lambda x : len(x[0]), reverse = True)
		comment_text, label, usr_id = zip(*batch)
		pad_comment_text = []
		lens = []
		max_len = len(comment_text[0])

		for i in range(len(label)):
			temp_label = [0] * max_len
			temp_label[:len(comment_text[i])] = comment_text[i]
			pad_comment_text.append(temp_label)
			lens.append(len(comment_text[i]))

		return pad_comment_text, label, lens, usr_id

def data_to_ids(train_data_path, train_data_id_path, vocab, stop_words_list_path):
	word_to_id = {}
	stop_words_list = json.load(open(stop_words_list_path, 'r'))
	for i in range(len(vocab)):
		word_to_id[vocab[i]] = i
	with open(train_data_path, 'r') as f_r:
		with open(train_data_id_path, 'w') as f_w:
			line = f_r.readline()
			while line:
				comment_text = re.split(r" +", line.strip().split("\t")[0])

				label = line.strip().split("\t")[-2]
				usr_id = line.strip().split("\t")[-1]
				word_ids = [word_to_id.get(word, -1) for word in comment_text if word not in stop_words_list and word in word_to_id]
				if (len(word_ids) == 0):
					f_w.write(str(1) + "\t" + str(label) + "\t" + str(usr_id) + "\n")
				else:
					f_w.write(" ".join([str(w) for w in word_ids]) + "\t" + str(label) + "\t" + str(usr_id) + "\n")
				line = f_r.readline()

if __name__ == '__main__':
	train_data_path = os.path.join(args.data_dir, "train_first.csv")
	train_data_splited_path = os.path.join(args.data_dir, "split_train_data.txt")
	train_data_id_path = os.path.join(args.data_dir, "train_data_ids.txt")
	predict_data_path = os.path.join(args.data_dir, "predict_data.csv")
	vocab_path = os.path.join(args.data_dir, "vocab")
	stop_words_path = os.path.join(args.data_dir, "stop_words.txt")
	stop_words_list_path = os.path.join(args.data_dir, "stop_words_list")

	predict_data_path = os.path.join(args.data_dir, "predict_first.csv")
	predict_data_splited_path = os.path.join(args.data_dir, "split_predict_data.txt")
	predict_data_id_path = os.path.join(args.data_dir, "predict_data_ids.txt")
	# comment_text, label = load_data(train_data_path, stop_words_path, vocab_path, stop_words_list_path)
	
	vocab = json.load(open(vocab_path, 'r'))
	print("vocab size is:" + str(len(vocab)))
	args.vocab_size = len(vocab)
	args.class_category = [1,2,3,4,5]
	args.kernel_sizes = [3,4,5] 
	args.batch_size = 1

	mode = "predict"
	# mode = "predict"
	if mode == "train":
		if not os.path.isfile(train_data_splited_path):
			slit_to_words(train_data_path, train_data_splited_path, mode)
		if not os.path.isfile(train_data_id_path):
			data_to_ids(train_data_splited_path, train_data_id_path, vocab, stop_words_list_path)
		dset = comment_dset(train_data_id_path, mode)
		data_loader = DataLoader(dset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
		cnn = model.CNN_Text(args)
		# train.train(data_loader, cnn, args)
		# args.model_path = "snapshot/snapshot_steps_7.pt"
		# cnn.load_state_dict(torch.load(args.model_path))
		train.train(data_loader, cnn, args)
	elif mode == "predict":
		if not os.path.isfile(predict_data_splited_path):
			slit_to_words(predict_data_path, predict_data_splited_path, mode)
		if not os.path.isfile(predict_data_id_path):
			data_to_ids(predict_data_splited_path, predict_data_id_path, vocab, stop_words_list_path)
		dset = comment_dset(predict_data_id_path, mode)
		data_loader = DataLoader(dset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
		cnn = model.CNN_Text(args)
		args.model_path = "snapshot/snapshot1_steps_7.pt"
		cnn.load_state_dict(torch.load(args.model_path))
		train.predict(data_loader, cnn, args)
	elif mode == "evaluate":
		if not os.path.isfile(train_data_splited_path):
			slit_to_words(train_data_path, train_data_splited_path, mode)
		if not os.path.isfile(train_data_id_path):
			data_to_ids(train_data_splited_path, train_data_id_path, vocab, stop_words_list_path)
		dset = comment_dset(train_data_id_path, mode)
		data_loader = DataLoader(dset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
		cnn = model.CNN_Text(args)
		# train.train(data_loader, cnn, args)
		for i in range(1, 17):
			args.model_path = "snapshot/snapshot1_steps_" + str(i) + ".pt"
			cnn.load_state_dict(torch.load(args.model_path))
			train.evaluate(data_loader, cnn, args, args.model_path, 0, 0)