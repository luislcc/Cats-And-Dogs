from models import BaseModel, basic_VGG, transfer_VGG16, transfer_ResNet, transfer_EfficientNet
import tensorflow as tf
from organizer import DirectoryOrganizer,ImageQuery
from models_utility import DataAugmentator
import pickle


#GPU
print(tf.config.list_physical_devices('GPU'))


#################################
### Meta Work ###
print("Organizing")
#data_cats_dogs = "CatsDogs"
#do = DirectoryOrganizer(data_cats_dogs,{"cat":1,"dog","pandas":1},sub_dirs={"train":1,"test":0.25,"val":0.15},workers=32)
#do.make()



data_cats_dogs_pandas = "CatsDogsPandas"
do = DirectoryOrganizer(data_cats_dogs_pandas,{"cat":1,"dog":1,"panda":1},train_folders=["trainWithPandas"],sub_dirs={"train":1,"test":0.25,"val":0.15},workers=32)
do.make()
##################################

print("Training Model")

##################################
### Testes ###

#DAugmentatorBasic = DataAugmentator(blockProb=0,noiseProb=0,contProb=0.15,brigProb=0.15,vFProb=0.25,hFProb=0.5,rotProb=0.25, trtProb=0.25)
DAugmentator = DataAugmentator()
DAugmentatorVal = DataAugmentator(contProb=0,brigProb=0,vFProb=0,hFProb=0,rotProb=0, trtProb=0)

CNN = BaseModel("VGG1Panda", basic_VGG,scale=255, blocks=1, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)


CNN = BaseModel("VGG1PandaDA", basic_VGG,scale=255, blocks=1, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("VGG2Panda", basic_VGG,scale=255, blocks=2, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("VGG2PandaDA", basic_VGG, scale=255, blocks=2, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("VGG3Panda", basic_VGG, scale=255, blocks=3, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)


CNN = BaseModel("VGG3PandaDA", basic_VGG, scale=255, blocks=3, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("VGG16Panda", transfer_VGG16, tf.keras.applications.vgg16.preprocess_input, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("VGG16PandaDA", transfer_VGG16, tf.keras.applications.vgg16.preprocess_input, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("ResNet50Panda", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("ResNet50PandaDA", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input, classes=3)

try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("EfficientNetB0Panda", transfer_EfficientNet, tf.keras.applications.efficientnet.preprocess_input, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal,save=False)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)


CNN = BaseModel("EfficientNetB0PandaDA", transfer_EfficientNet, tf.keras.applications.efficientnet.preprocess_input, classes=3)
try:
	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal,save=False)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)


###################################