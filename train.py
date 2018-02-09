import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import csv
from torch.autograd import Variable

corrects = 0
example_num = 0

def train(train_iter, model, args):
	if args.no_cuda:
		model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

	steps = 0
	best_acc = 0
	last_step = 0
	model.train()
	for epoch in range(1, args.epochs+1):
		for batch in train_iter:
			feature, target, lens, usr_id = batch
			# print(feature)
			# print(target)
			# print (len(feature))
			# print (len(feature[0]))
			feature = Variable(torch.LongTensor(feature), requires_grad = False)
			target = Variable(torch.LongTensor(target), requires_grad = False)
			if args.no_cuda:
				feature, target = feature.cuda(), target.cuda()

			optimizer.zero_grad()
			logit = model(feature)
			loss = F.cross_entropy(logit, target)
			sorted_score, indices = torch.sort(logit, descending = True)
			# print(indices[:,0].data + 1)
			# print(target.data + 1)
			# print ("logit is: " + str(logit))
			# print ("target is: " + str(target))
			
			# loss_func = torch.nn.NLLLoss()
			# loss = -loss_func(logit, target)
			# print (logit.data)
			# print (target.data)
			# print (loss.data)
			# print ("steps_{}_loss is: ".format(steps, loss.data[0])
			loss.backward()
			optimizer.step()
			steps += 1
			if steps % args.log_interval == 0:
				corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
				accuracy = 100 * corrects/args.batch_size
				sys.stdout.write(
					'\rBatch[{}] - loss: {:.6f} acc: {:.4f}%({}/{})'.format(steps,
																			loss.data[0],
																			accuracy,
																			corrects,
																			args.batch_size)
					)
			if steps % args.save_interval == 0:
				save(model, args.save_dir, 'snapshot2', epoch, steps)

def predict(predict_iter, model, args):
	csvfile = open("./data/predict_result.csv", "w")
	writer = csv.writer(csvfile)
	data = []
	scores = []
	for batch in predict_iter:
		feature, label, lens, usr_id = batch
		feature = Variable(torch.LongTensor(feature), requires_grad = False)

		score = model(feature)
		sorted_score, indices = torch.sort(score, descending = True)
		print(indices[0][0].data[0] + 1)
		data.append([usr_id[0], indices[0][0].data[0] + 1])

		scores.append(score)
	writer.writerows(data)
	return scores

def evaluate(evaluate_iter, model, args, model_name, corrects, example_num):
	csvfile = open("./data/evaluate_result.csv", "a")
	writer = csv.writer(csvfile)
	data = []
	scores = []
	for batch in evaluate_iter:
		feature, target, lens, usr_id = batch
		feature = Variable(torch.LongTensor(feature), requires_grad = False)
		target = Variable(torch.LongTensor(target), requires_grad = False)
		logit = model(feature)
		corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
		example_num += args.batch_size
		accuracy = corrects/example_num

		if (example_num % 50000 == 0):
			print("current true is {}, current example num is {}, accuracy is {}"
				.format(corrects, example_num, accuracy))
	data.append([model_name, accuracy])
	writer.writerows(data)
	return scores

def save(model, save_dir, save_prefix, epoch, steps):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_prefix = os.path.join(save_dir, save_prefix)
	save_path = '{}_steps_{}.pt'.format(save_prefix, epoch, steps)
	torch.save(model.state_dict(), save_path)