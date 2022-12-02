from models import BaseModel, basic_VGG, transfer_VGG16, transfer_ResNet, transfer_EfficientNet
import tensorflow as tf
from organizer import DirectoryOrganizer,ImageQuery
from models_utility import DataAugmentator
import pickle


#GPU
print(tf.config.list_physical_devices('GPU'))


#################################
### Meta Work ###

data_cats_dogs = "CatsDogs"
#final_cats_dogs = "CatsDogsFinal"
#do = DirectoryOrganizer(data_cats_dogs,["cat","dog"],seed=1, val_ratio=0.25,workers=32)
#do.make()
#doFinal = DirectoryOrganizer(final_cats_dogs,["cat","dog"],seed=1, val_ratio=0.0,workers=32)
#doFinal.make()


#data_cats_dogs_pandas = "CatsDogsPandas"
#final_cats_dogs_pandas = "CatsDogsPandasFinal"
#iq = ImageQuery("train","https://api.cognitive.microsoft.com/bing/v7.0/images/search")
#iq.query("panda")
#do = DirectoryOrganizer(data_cats_dogs_pandas,["cat","dog","panda"],seed=1, val_ratio=0.25,workers=32)
#do.make()
#do = DirectoryOrganizer(final_cats_dogs_pandas,["cat","dog","panda"],seed=1, val_ratio=0.0,workers=32)
#do.make()
##################################

print("Training Model")

##################################
### Testes ###

#DAugmentatorBasic = DataAugmentator(blockProb=0,noiseProb=0,contProb=0.15,brigProb=0.15,vFProb=0.25,hFProb=0.5,rotProb=0.25, trtProb=0.25)
DAugmentator = DataAugmentator()
DAugmentatorVal = DataAugmentator(contProb=0,brigProb=0,vFProb=0,hFProb=0,rotProb=0, trtProb=0)

#CNN = BaseModel("VGG1", basic_VGG,scale=255, blocks=1)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)


#CNN = BaseModel("VGG1DA", basic_VGG,scale=255, blocks=1)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)



#CNN = BaseModel("VGG2", basic_VGG,scale=255, blocks=2)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)



#CNN = BaseModel("VGG2DA", basic_VGG, scale=255, blocks=2)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)



#CNN = BaseModel("VGG3", basic_VGG, scale=255, blocks=3)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)


#CNN = BaseModel("VGG3DA", basic_VGG, scale=255, blocks=3)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
#except waitTimeException as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)



CNN = BaseModel("VGG16", transfer_VGG16, tf.keras.applications.vgg16.preprocess_input)
try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("VGG16DA", transfer_VGG16, tf.keras.applications.vgg16.preprocess_input)
try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("ResNet50", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input)
try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("ResNet50DA", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input)

try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("EfficientNetB0", transfer_EfficientNet, tf.keras.applications.efficientnet.preprocess_input)
try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal,save=False)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)


CNN = BaseModel("EfficientNetB0DA", transfer_EfficientNet, tf.keras.applications.efficientnet.preprocess_input)
try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal,save=False)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)


###################################