from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.optimizers import SGD, Adam

from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB0

from keras.callbacks import EarlyStopping

from matplotlib import pyplot
from keras.models import Model

import tensorflow as tf
from models_utility import DataAugmentator,DataFlowClasses
import os
import json



def basic_VGG(blocks=1, classes=2, hiddenDefs=[(128,"relu")], input_shape=(200, 200, 3)):
	model = Sequential()
	
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
	model.add(MaxPooling2D((2, 2)))

	for block in range(1,blocks-1):
		model.add(Conv2D(64*block, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))	

	model.add(Flatten())
	
	for neuronNr,actv in hiddenDefs:
		model.add(Dense(neuronNr, activation=actv, kernel_initializer='he_uniform'))
	
	model.add(Dense(classes, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
	return model




def tutorial_transfer(arch):
	def __define_model(classes=2,hiddenDefs=[(128,"relu")],input_shape=(200, 200, 3)):
		# load model
		model = arch(include_top=False, weights="imagenet",input_shape=input_shape)
		# mark loaded layers as not trainable
		for layer in model.layers:
			layer.trainable = False
		
		# add new classifier layers
		flat1 = Flatten()(model.layers[-1].output)
		auxHdn = flat1
		for neuronNr,actv in hiddenDefs:
			auxHdn = Dense(neuronNr, activation=actv, kernel_initializer='he_uniform')(auxHdn)
		
		output = Dense(classes, activation='softmax')(auxHdn)
		# define new model
		model = Model(inputs=model.inputs, outputs=output)
		# compile model
		optimizer = SGD(learning_rate=0.001, momentum=0.9)
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
		return model

	return __define_model



transfer_VGG16 = tutorial_transfer(VGG16)
transfer_ResNet = tutorial_transfer(ResNet50)
transfer_EfficientNet = tutorial_transfer(EfficientNetB0)





class BaseModel(object):
	def __init__(self, name, modelFunction,preProcessInput=None, scale=1,**model_opts):
		self.name = name
		self.model = modelFunction(**model_opts)
		self.history = None
		self.scale = scale
		self.model_opts = model_opts
		self.modelFunction = modelFunction
		if preProcessInput is None:
			self.preProcessInput = lambda x: x
		else:
			self.preProcessInput = preProcessInput
		pass

	def reset_model(self):
		self.model = modelFunction(**(self.model_opts))

	def run_test_harness(self,dataSetFolder,classes, dataAugmentatorTrain,dataAugmentatorValid,epochs=20,verbose=1,save=True):
		train_dir = os.path.join(dataSetFolder,"train","")
		val_dir = os.path.join(dataSetFolder,"val","")

		train_it = DataFlowClasses(train_dir,classes,[dataAugmentatorTrain.processImage, self.preProcessInput],scale=self.scale,bufferWorkers=16,batch_size=32)
		val_it = DataFlowClasses(val_dir,classes,[dataAugmentatorValid.processImage, self.preProcessInput],scale=self.scale,bufferWorkers=16,batch_size=32)
		
		callback = EarlyStopping(monitor='val_loss', patience=3)
		self.history = self.model.fit(train_it, steps_per_epoch=len(train_it), callbacks=[callback], validation_data=val_it, validation_steps=len(val_it), epochs=epochs, verbose=verbose)
		evaluation = self.model.evaluate(val_it, steps=len(val_it), verbose=0)
		
		if save:
			self.model.save(self.name + '_Train.h5')
			print(f"model {self.name} Evaluation:")
			print(evaluation)
	

	def summarize(self,file_name=None):
		assert self.history is not None
		file_name = file_name or self.name
		
		with open(f'{file_name}.json', 'w') as outp:
			json.dump(self.history.history, outp)
		pass	