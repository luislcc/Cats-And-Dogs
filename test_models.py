from models import BaseModel, basic_VGG, transfer_VGG16, transfer_ResNet, transfer_EfficientNet
import tensorflow as tf
from organizer import DirectoryOrganizer,ImageQuery
from models_utility import DataAugmentator
import pickle
import os

#GPU
print(tf.config.list_physical_devices('GPU'))


#################################
### Meta Work ###
#print("Organizing")
#do = DirectoryOrganizer(data_cats_dogs,{"cat":1,"dog","pandas":1},sub_dirs={"train":1,"test":0.25,"val":0.15},workers=32)
#do.make()


#


#do = DirectoryOrganizer(data_cats_dogs_pandas,{"cat":0.3,"dog":0.3,"panda":1},train_folders=["trainWithPandas"],sub_dirs={"train":1,"test":0.25,"val":0.15},workers=32)
#do.make()



data_cats_dogs = "CatsDogs"
data_cats_dogs_pandas = "CatsDogsPandas"
data_cats_dogs_pandas_unbalance = "CatsDogsPandasUnbalanced"

testFolderUnbalance = os.path.join(data_cats_dogs_pandas,"test","")

DAugmentator = DataAugmentator()
DAugmentatorVal = DataAugmentator(contProb=0,brigProb=0,vFProb=0,hFProb=0,rotProb=0, trtProb=0)


#CNN = BaseModel("VGG1", basic_VGG,scale=255, blocks=1, classes=2)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("VGG1DA", basic_VGG,scale=255, blocks=1, classes=2)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
##################################
#
#
#CNN = BaseModel("VGG1Panda", basic_VGG,scale=255, blocks=1, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("VGG1PandaDA", basic_VGG,scale=255, blocks=1, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
#################################
#
#
#
#CNN = BaseModel("VGG1PandaUnbalance", basic_VGG,scale=255, blocks=1, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("VGG1PandaUnbalanceDA", basic_VGG,scale=255, blocks=1, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentator,DAugmentatorVal,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#################################
#
#
#
#
#CNN = BaseModel("VGG2", basic_VGG,scale=255, blocks=2, classes=2)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
#CNN = BaseModel("VGG2DA", basic_VGG, scale=255, blocks=2, classes=2)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
############################
#
#
#CNN = BaseModel("VGG2Panda", basic_VGG,scale=255, blocks=2, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("VGG2PandaDA", basic_VGG, scale=255, blocks=2, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal)
#except Exception as e:
#	print(e)
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#############################
#
#
#
#CNN = BaseModel("VGG2PandaUnbalance", basic_VGG,scale=255, blocks=2, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("VGG2PandaUnbalanceDA", basic_VGG, scale=255, blocks=2, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentator,DAugmentatorVal,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
#
###################################
#
#CNN = BaseModel("VGG3", basic_VGG, scale=255, blocks=3, classes=2)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("VGG3DA", basic_VGG, scale=255, blocks=3, classes=2)
#try:
#	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
#######################################
#
#
#CNN = BaseModel("VGG3Panda", basic_VGG, scale=255, blocks=3, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("VGG3PandaDA", basic_VGG, scale=255, blocks=3, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#######################################
#
#
#CNN = BaseModel("VGG3PandaUnbalanced", basic_VGG, scale=255, blocks=3, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("VGG3PandaUnbalancedDA", basic_VGG, scale=255, blocks=3, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentator,DAugmentatorVal,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
###################################



CNN = BaseModel("VGG16", transfer_VGG16, tf.keras.applications.vgg16.preprocess_input, classes=2)
try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("VGG16DA", transfer_VGG16, tf.keras.applications.vgg16.preprocess_input, classes=2)
try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)


###################################


#CNN = BaseModel("VGG16Panda", transfer_VGG16, tf.keras.applications.vgg16.preprocess_input, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("VGG16PandaDA", transfer_VGG16, tf.keras.applications.vgg16.preprocess_input, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)




###################################

#CNN = BaseModel("VGG16PandaUnbalanced", transfer_VGG16, tf.keras.applications.vgg16.preprocess_input, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
#CNN = BaseModel("VGG16PandaUnbalancedDA", transfer_VGG16, tf.keras.applications.vgg16.preprocess_input, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentator,DAugmentatorVal,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#


################################



CNN = BaseModel("ResNet50", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input, classes=2)
try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



CNN = BaseModel("ResNet50DA", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input, classes=2)

try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)


################################



#CNN = BaseModel("ResNet50Panda", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
#CNN = BaseModel("ResNet50PandaDA", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input, classes=3)
#
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
#################################
#
#
#CNN = BaseModel("ResNet50PandaUnbalanced", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
#CNN = BaseModel("ResNet50PandaUnbalancedDA", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input, classes=3)
#
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentator,DAugmentatorVal,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#

##################################



CNN = BaseModel("EfficientNetB0", transfer_EfficientNet, tf.keras.applications.efficientnet.preprocess_input, classes=2)
try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentatorVal,DAugmentatorVal,save=False)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)


CNN = BaseModel("EfficientNetB0DA", transfer_EfficientNet, tf.keras.applications.efficientnet.preprocess_input, classes=2)
try:
	CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal,save=False)
except Exception as e:
	print(e)

try:
	CNN.summarize()
except Exception as e:
	print(e)



###################################
#
#
#
#CNN = BaseModel("EfficientNetB0Panda", transfer_EfficientNet, tf.keras.applications.efficientnet.preprocess_input, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal,save=False)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("EfficientNetB0PandaDA", transfer_EfficientNet, tf.keras.applications.efficientnet.preprocess_input, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas,["cat","dog","panda"],DAugmentator,DAugmentatorVal,save=False)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#
#
#
#
####################################
#
#
#CNN = BaseModel("EfficientNetB0PandaUnbalanced", transfer_EfficientNet, tf.keras.applications.efficientnet.preprocess_input, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentatorVal,DAugmentatorVal,save=False,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)
#
#
#CNN = BaseModel("EfficientNetB0PandaUnbalancedDA", transfer_EfficientNet, tf.keras.applications.efficientnet.preprocess_input, classes=3)
#try:
#	CNN.run_test_harness(data_cats_dogs_pandas_unbalance,["cat","dog","panda"],DAugmentator,DAugmentatorVal,save=False,testSetFolder=testFolderUnbalance)
#except Exception as e:
#	print(e)
#
#try:
#	CNN.summarize()
#except Exception as e:
#	print(e)