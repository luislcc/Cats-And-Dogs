import cv2
import glob
import os
import threading
import math
import queue
import random
import numpy as np
import copy
from keras.utils import Sequence
import tensorflow as tf

class DataAugmentator(object):
	def __init__(self,shape=(200,200),blockSizeRange=(40,90),noiseVariability=0.15, contRange=(0.3,1.7), brigRange=(-127,127),
				blockProb=0.5,noiseProb=0.5,degRange=(-180,180),contProb=0.15,brigProb=0.15,vFProb=0.25,hFProb=0.5,rotProb=0.25, trtProb=0.25,
				rescale=True):
		self.shape = shape
		self.blockSizeRange = blockSizeRange
		self.noiseVariability = noiseVariability
		self.blockProb = blockProb
		self.noiseProb = noiseProb
		self.degRange = degRange
		self.contProb = contProb
		self.brigProb = brigProb
		self.vFProb = vFProb
		self.hFProb = hFProb
		self.rotProb = rotProb
		self.brigRange = brigRange
		self.contRange = contRange
		self.trtProb = trtProb
		self.rescale = rescale
	
	
	def randomTranslate(self,img):
		if random.random() < self.trtProb:
			height, width = img.shape[:2]
			quarter_height, quarter_width = int(height / 4), int(width / 4)
			trH = np.random.randint(-quarter_height,quarter_height)
			trW = np.random.randint(-quarter_width,quarter_width)
			T = np.float32([[1, 0, trH], [0, 1, trW]])
			img = cv2.warpAffine(img, T, (width, height))
		return img


	def randomContrastBrightness(self,img):
		alpha = 1
		beta = 0
		if random.random() < self.contProb:
			alpha = np.random.uniform(*self.contRange)

		if random.random() < self.brigProb:
			beta = np.random.randint(*self.brigRange)

		return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)/255

	
	def randomFlip(self,img):
		if random.random() < self.vFProb:
			img = cv2.flip(img, 0)

		if random.random() < self.hFProb:
			img = cv2.flip(img, 1)
		
		return img
	

	def randomRotation(self,img):
		if random.random() < self.rotProb:
			shp = img.shape
			degs = np.random.randint(*self.degRange)
			M = cv2.getRotationMatrix2D((shp[0]/2,shp[1]/2),degs,1) 
			img = cv2.warpAffine(img,M,(shp[0],shp[1])) 
		
		return img


	def randomBlockingArtifact(self,img):
		if random.random() < self.blockProb:
			shp = img.shape
			x = np.random.triangular(0,shp[0]/2,shp[0])
			y = np.random.triangular(0,shp[1]/2,shp[1])
			x = math.ceil(x)
			y = math.ceil(y)
			dx = np.random.randint(*self.blockSizeRange)
			
			if random.random() > 0.5: 
				img[x:x+dx,y:y+dx,:] = 0

			else:
				img = cv2.circle(img,(x,y),dx,(0,0,0),-1)
		return img

	def randomNoise(self,img):
		if random.random() < self.noiseProb:
			deviation = self.noiseVariability*random.random()
			noise = np.random.normal(0, deviation, img.shape)
			img += noise
			np.clip(img, 0., 1.)
		return img


	def processImage(self,image):
		res = image
		res = cv2.resize(res, self.shape)
		
		res = self.randomContrastBrightness(res)
		res = self.randomNoise(res) 
		res = self.randomBlockingArtifact(res)
		res = self.randomTranslate(res)
		res = self.randomFlip(res)
		res = self.randomRotation(res)

		if self.rescale:
			res = res*255

		return res
	



class DataFlowClasses(Sequence):
	def __init__(self, direc, classes, preprocess, batch_size=16, bufferImages=34, bufferReadMax=64, bufferWorkers=16):
		self.classesDir = {clss:os.path.join(direc,clss) for clss in classes}
		self.dataPreProcessor = preprocess
		self.batch_size = batch_size
		
		
		self.images = [(i,os.path.join(self.classesDir[x],y)) for i,x in enumerate(classes) for y in os.listdir(self.classesDir[x]) if (y.endswith(".jpg") or y.endswith(".png"))]
		self.remainingImgs = []
		self.totalImages = len(self.images)
		self.classNumber = len(classes) - 1

		self.readerThr = None
		self.imageThrs = []

		self.bufferImagesSz = bufferImages
		self.bufferReadSz = bufferReadMax
		
		self.workers = bufferWorkers
		self.bufferRead = None
		self.bufferImages = None

		self.lstImage = None
		self.lstCateg = None

		self.on_epoch_end()
		pass

	def __len__(self):
		return int(np.floor(self.totalImages / float(self.batch_size)))


	def __thrd_readImages(self):
		for clss,imgPath in self.remainingImgs:
			img = cv2.imread(imgPath)
			self.bufferRead.put((clss,img))
		return

	
	def __thrd_preProcess(self,imgAmmount):
		for _ in range(imgAmmount):
			clss,img = self.bufferRead.get()
			
			oneHot = np.zeros(self.classNumber+1)
			oneHot[clss] = 1

			imgPP = img
			for pp in self.dataPreProcessor:
				imgPP = pp(imgPP)
			
			value = (oneHot,imgPP)
			self.bufferImages.put(value)

			self.bufferRead.task_done()
		self.bufferRead.join()
		return


	def __startBuffer(self):
		if self.readerThr:
			self.readerThr.join()

		for thr in self.imageThrs:
			thr.join()

		self.bufferRead = queue.Queue(maxsize=self.bufferReadSz)
		self.bufferImages = queue.Queue(maxsize=self.bufferImagesSz)

		threadsImageAmmount = [math.floor(self.totalImages/self.workers) for x in range(self.workers)]

		additional = self.totalImages - sum(threadsImageAmmount)

		for p in range(additional):
			threadsImageAmmount[p] += 1

		self.readerThr = threading.Thread(target=self.__thrd_readImages,args=(),daemon=True)	
		self.imageThrs = [threading.Thread(target=self.__thrd_preProcess,args=(threadsImageAmmount[i],),daemon=True) for i in range(self.workers)]
		self.readerThr.start()
		for i in self.imageThrs:
			i.start()
		pass



	def on_epoch_end(self):
		self.remainingImgs = [x for x in self.images]
		random.shuffle(self.remainingImgs)
		self.__startBuffer()


	def __getitem__(self, index):
		X,y = [],[]
		
		for _ in range(self.batch_size):
			try:
				oneHot,image = self.bufferImages.get(timeout=1.0)
			
			except Exception as e:
				oneHot,image = self.lstCateg,self.lstImage
			
			self.lstImage = image
			self.lstCateg = oneHot

			X.append(image)
			y.append(oneHot)

		return np.array(X), np.array(y)



if __name__ == '__main__':
	DA = DataAugmentator(rescale=True)
	test = DataFlowClasses(r"D:\Mestrado\TopicosAI\Projeto\Cats-And-Dogs\CatsDogs\train",["cat","dog"],[DA.processImage,tf.keras.applications.resnet50.preprocess_input],bufferWorkers=1)

	c = 0
	print(len(test))
	for t in range(len(test)):
		X,y = test[t]
		#print(X.shape)
		#print(y.shape)
		for i,val in enumerate(X):
			print(val.shape)
			print(y[i])
			cv2.imshow("bruh",val/255)
			#print(msk.shape)
			#print(np.unique(msk))
			cv2.waitKey(0)
			c += 1

	print(c)

