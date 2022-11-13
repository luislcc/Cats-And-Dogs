import os
from shutil import copyfile
from random import seed
from random import random
import threading
import math
import requests
import cv2
from dotenv import load_dotenv


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
			seed(self.seed)
		
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



class ImageQuery(object):
	def __init__(self, download_folder, API_URL, dotenv_file=None):
		self.dotenv_file = dotenv_file
		self.download_folder = download_folder
		self.API_URL = API_URL


	def query(self,keyword,group_size=50,max_results=250):
		if self.dotenv_file is not None:
			load_dotenv(self.dotenv_file)
		else:
			load_dotenv()
		headers = {"Ocp-Apim-Subscription-Key" : os.getenv('API_KEY')}
		params = {"q": keyword, "offset": 0, "count": group_size}
		count = 0
		
		search = requests.get(self.API_URL, headers=headers, params=params)
		search.raise_for_status()
		results = search.json()
		
		estNumResults = min(results["totalEstimatedMatches"], max_results)
		
		for offset in range(0, estNumResults, group_size):
			params["offset"] = offset
			search = requests.get(self.API_URL, headers=headers, params=params)
			search.raise_for_status()
			results = search.json()
		
			for v in results["value"]:
				# try to download the image
				try:
					# make a request to download the image
					r = requests.get(v["contentUrl"], timeout=30)
					# build the path to the output image
					ext = v["contentUrl"][v["contentUrl"].rfind("."):]
					p = os.path.join(self.download_folder, f"{keywords.replace(' ','.')}.{str(count).zfill(8)}{ext}")
					# write the image to disk
					f = open(p, "wb")
					f.write(r.content)
					f.close()

					image = cv2.imread(p)
					if image is None:
						os.remove(p)
						continue

					else:
						total += 1	
				
				except Exception as e:
					continue
			




