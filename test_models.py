from models import basic_VGG,transfer_VGG16
import tensorflow as tf
from organizer import DirectoryOrganizer


def summarize(history,plot_name):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	pyplot.savefig(plot_name + '_plot.png')
	pyplot.close()
	pass


def run_test_harness(model,summarize_plot_name,train_directory,test_directory):
	# create data generator
	
	# v Data augmentation goes here v 
	train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	
	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
	
	# prepare iterators
	train_it = train_datagen.flow_from_directory(train_directory,class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = test_datagen.flow_from_directory(test_directory,class_mode='binary', batch_size=64, target_size=(200, 200))
	
	# fit model
	history = model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize(history,summarize_plot_name)



def final_test_harness(model,model_name,train_directory):
	# Wtf is going on?
	# create data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	datagen.mean = [123.68, 116.779, 103.939]
	
	# prepare iterator
	train_it = datagen.flow_from_directory(train_directory,class_mode='binary', batch_size=64, target_size=(224, 224))
	# fit model
	model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=0)
	
	# save model
	model.save(model_name + '.h5')



#do = DirectoryOrganizer("CatsDogs",["cat","dog"],seed=1, val_ratio=0.25,workers=32)
#do.make()

#do = DirectoryOrganizer("CatsDogsPandas",["cat","dog","panda"],seed=1, val_ratio=0.25,workers=32,train_folders=["train","trainPandas"])
#do.make()

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


VGG1 = basic_VGG(1)
run_test_harness(VGG1,"VGG1",,)
final_test_harness(VGG1,"VGG1",)


VGG2 = basic_VGG(2)
run_test_harness(VGG2,"VGG2",,)
final_test_harness(VGG2,"VGG2",)s

VGG3 = basic_VGG(3)
run_test_harness(VGG3,"VGG3",,)
final_test_harness(VGG3,"VGG3",)


VGG16 = transfer_VGG16()
run_test_harness(VGG16,"VGG16",,)
final_test_harness(VGG16,"VGG16",)


# make sure it runs on a GPU
# make data augmentation
# make models for +2 elements dataSet


