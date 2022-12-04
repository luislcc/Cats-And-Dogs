import os
from shutil import copyfile
from random import random
import threading
import math
import requests
import cv2
from dotenv import load_dotenv


class DirectoryOrganizer(object):
	def __init__(self, new_folder,labels,train_folders=["train"],sub_dirs={"train":1,"val":0.1,"test":0.1},dataset_folder="", workers=16):
		self.new_folder = new_folder
		self.dataset_folder = dataset_folder
		
		if dataset_folder:
			self.train_folders = [os.path.join(dataset_folder,train_folder,"") for train_folder in train_folders]
		else:
			self.train_folders = [os.path.join(train_folder,"") for train_folder in train_folders]
		
		self.labels = labels
		self.sub_dirs = list(sub_dirs.keys())
		self.ratios = [(x,sum(list(sub_dirs.values())[i:])) for i,x in enumerate(sub_dirs.keys())]
		self.folders = {}

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
		
		for folder in folders:
			foldDir = os.listdir(folder)
			img_per_thrd = math.ceil(len(foldDir)/self.workers)
			
			lsDir = [foldDir[i:i+img_per_thrd] for i in range(0,len(foldDir),img_per_thrd)]
			def copyAux(index):
				for file in lsDir[index]:
					src = os.path.join(folder,file)
					
					dst_dir = 'train'
					pickedValue = random()
					for sub_dir,odd in self.ratios:
						if pickedValue < odd:
							dst_dir = sub_dir

						else:
							break
						
					for label in self.labels:
						if file.startswith(label):
							if random() < self.labels[label]:
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
	def __init__(self, download_folder, dotenv_file=None):
		self.dotenv_file = dotenv_file
		self.download_folder = download_folder


	def query(self,keyword,group_size=100,groups=10):
		if self.dotenv_file is not None:
			load_dotenv(self.dotenv_file)
		else:
			load_dotenv()

		

		params = {
			"api_key": os.getenv('API_KEY'),
			"device": "desktop",
			"engine": "google",
			"q": keyword,
			"location": "Austin, Texas, United States",
			"google_domain": "google.com",
			"gl": "us",
			"hl": "en",
			"tbm": "isch",
			"ijn": 0 #nosso offset
		}

		image_results = []
		search = GoogleSearch(params)
		count = 0

		


		for offset in range(0, estNumResults, group_size):
			try:
				results = search.get_dict()
				
				if "error" not in results:
					for image in results["images_results"]:
						if image["original"] not in image_results:
							image_results.append(image["original"])
			
					# update to the next page
					params["ijn"] += 1
				
				else:
					images_is_present = False
					#print(results["error"])
			
			except Exception as e:
				continue

		
		for v in image_results:
			# try to download the image
			try:
				# make a request to download the image
				r = requests.get(v, timeout=30)
				# build the path to the output image
				ext = v[v.rfind("."):]
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
					count += 1

			except Exception as e:
				continue
			

			




