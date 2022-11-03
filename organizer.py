import os
from shutil import copyfile
from random import seed
from random import random
import threading
import math 


class DirectoryOrganizer(object):
	def __init__(self, new_folder,labels,train_folders=["train"],dataset_folder="",seed=None, val_ratio=0.25, workers=16):
		self.new_folder = new_folder
		self.dataset_folder = dataset_folder
		
		if dataset_folder:
			self.train_folders = [os.path.join(dataset_folder,train_folder,"") for train_folder in train_folders]
		else:
			self.train_folders = [os.path.join(train_folder,"") for train_folder in train_folders]
		
		
		self.labels = labels
		self.sub_dirs = ["train","val"]
		self.folders = {}
		
		self.seed = None
		if seed is not None:
			self.seed=seed
		
		self.val_ratio=val_ratio
		self.workers = workers
		pass

	def make_structure(self):
		os.makedirs(os.path.join(self.new_folder,""), exist_ok=True)
		for label in self.labels:
			self.folders[label] = {}
			for sub_dir in self.sub_dirs:
				newdir = os.path.join(self.new_folder,sub_dir,label,"")
				self.folders[label][sub_dir] = newdir
				os.makedirs(newdir, exist_ok=True)
		pass

	def make(self):
		self.make_structure()
		folders = self.train_folders
		if self.seed is not None:
			seed(1)
		
		for folder in folders:
			foldDir = os.listdir(folder)
			img_per_thrd = math.ceil(len(foldDir)/self.workers)
			
			lsDir = [foldDir[i:i+img_per_thrd] for i in range(0,len(foldDir),img_per_thrd)]
			def copyAux(index):
				for file in lsDir[index]:
					src = os.path.join(folder,file)
					
					dst_dir = 'train'
					if random() < self.val_ratio:
						dst_dir = 'val'
					
					for label in self.labels:
						if file.startswith(label):
							dest = os.path.join(self.folders[label][dst_dir],file)
							copyfile(src, dest)
							break
				pass

			threads = [threading.Thread(target=copyAux,args=(i,)) for i in range(self.workers)]
			for i in range(self.workers):
				threads[i].start()

			for i in range(self.workers):
				threads[i].join()
		pass

