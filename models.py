from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot




class BaseModel(object):
	def __init__(self, name, model):
		self.name = name
		self.model = model
		self.history = None

	def run_test_harness(self,dataSetFolder,data_augmentation={}):
		train_datagen = ImageDataGenerator(rescale=1.0/255.0, **data_augmentation)
		test_datagen = ImageDataGenerator(rescale=1.0/255.0)

		train_dir = os.path.join(dataSetFolder,"train","")
		val_dir = os.path.join(dataSetFolder,"val","")
		train_it = train_datagen.flow_from_directory(train_dir, batch_size=64, target_size=(200, 200))
		test_it = test_datagen.flow_from_directory(val_dir, batch_size=64, target_size=(200, 200))
		
		self.history = self.model.fit(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
		_, acc = self.model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
		print(f"model {self.name} accuracy:")
		print('> %.3f' % (acc * 100.0))

	def final_test_harness(self,dataSetFolder):
		datagen = ImageDataGenerator(featurewise_center=True)#?
		datagen.mean = [123.68, 116.779, 103.939]#?
		
		# prepare iterator
		train_dir = os.path.join(dataSetFolder,"train","")
		train_it = datagen.flow_from_directory(train_dir, batch_size=64, target_size=(224, 224))
		self.model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=0)
		self.model.save(self.name + '.h5')

	def summarize(self,plot_name=None):
		assert self.history is not None
		# plot loss
		pyplot.subplot(211)
		pyplot.title('Loss')
		pyplot.plot(self.history.history['loss'], color='blue', label='train')
		pyplot.plot(self.history.history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(212)
		pyplot.title('Classification Accuracy')
		pyplot.plot(self.history.history['accuracy'], color='blue', label='train')
		pyplot.plot(self.history.history['val_accuracy'], color='orange', label='test')
		# save plot to file
		plot_name = plot_name or self.name
		pyplot.savefig(plot_name + '_plot.png')
		pyplot.close()
		pass	




def basic_VGG(blocks=1):
	blocks = min(blocks,3)
	model = Sequential()
	
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))

	for block in range(1,blocks-1):
		model.add(Conv2D(64*block, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D((2, 2)))	

	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model




def transfer_VGG16():
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model