from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class comment_dset(Dataset):
	def __init__(self, comment_path, mode):
		with open(comment_path, 'r') as f:
			self.comment_list = []
			self.label_list = []
			self.usr_id_list = []
			lines = f.readlines()
			if (mode == "train"):
				lines = sorted(lines, key = lambda l : len(l), reverse = True)
			for i in lines:
				self.comment_list.append([int(k) for k in i.split("\t")[0].split(" ")])
				self.label_list.append(int(i.split("\t")[-2].strip()) - 1)
				self.usr_id_list.append(str(i.split("\t")[-1].strip()))
			# self.comment_list = [[int(k) for k in i.split("***")[0].split(" ")] for i in lines]
			# self.label_list = [(int(i.split("***")[1].strip()) - 1) for i in lines]
			
			print (len(self.comment_list))
			print (len(self.label_list))
			print (len(self.usr_id_list))
	def __getitem__(self, index):
		comment_text = self.comment_list[index]
		label = self.label_list[index]
		usr_id_list = self.usr_id_list[index]
		return comment_text, label, usr_id_list

	def __len__(self):
		return len(self.label_list)

