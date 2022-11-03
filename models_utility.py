import random
import numpy as np


class DataAugmentator(object):
	def __init__(self, **kwargs):
		self.width_shift_range= 0.3
		self.height_shift_range= 0.3
		self.horizontal_flip= True
		self.rotation_range= 90
		self.zoom_range= [0.5,1.5]
		self.noise = True
		self.__dict__.update(**kwargs)

	def add_noise(img):
		if random.random() > 0.5:
			VARIABILITY = 50
			deviation = VARIABILITY*random.random()
			noise = np.random.normal(0, deviation, img.shape)
			img += noise
			np.clip(img, 0., 255.)
		return img

	def values(self):
		res = self.__dict__
		if res["noise"]:
			res["preprocessing_function"] = DataAugmentor.add_noise

		del res["noise"]

		return res
	
