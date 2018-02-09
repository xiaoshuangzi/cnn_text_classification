import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_Text(nn.Module):
	def __init__(self, args):
		super(CNN_Text, self).__init__()
		self.args = args

		self.vocab_size = args.vocab_size
		self.embed_dim = args.embed_dim

		self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
		self.softmax = nn.Softmax()
		# self.convs1 = nn.Sequential(
		# 				nn.Conv2d(1, args.kernel_num, args.kernel_sizes[0])，
		# 				nn.ReLU(),
		# 				nn.MaxPool2d(2),
		# 			)
		# self.convs2 = nn.Sequential(
		# 				nn.Conv2d(args.kernel_num, args.kernel_num*2, args.kernel_sizes[1]),
		# 				nn.RelU(),
		# 				nn.MaxPool2d(2),
		# 			)
		Ci = 1
		Co = args.kernel_num
		Ks = args.kernel_sizes
		V = args.vocab_size
		D = args.embed_dim
		C = len(args.class_category)
		# 输入通道，输出通道（小朋友数），（过滤窗口size，此处为3，4，5 and embeding维度，就是每次看几个词）
		self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), padding = (K, 0)) for K in Ks])

		self.dropout = nn.Dropout(args.dropout)
		self.fc1 = nn.Linear(len(Ks)*Co, C)

	def conv_and_pool(self, x, conv):
		x = F.relu(conv(x)).squeeze(3)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		return x

	def forward(self, x):
		x = self.embed(x)
		if self.args.static:
			x = Variable(x)
		x = x.unsqueeze(1)


		# x0 = F.relu(self.convs1[0](x)).squeeze(3)
		# x1 = F.relu(self.convs1[1](x)).squeeze(3)
		# x2 = F.relu(self.convs1[2](x)).squeeze(3)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

		x = torch.cat(x, 1)

		x = self.dropout(x)

		logit = self.fc1(x)

		return logit