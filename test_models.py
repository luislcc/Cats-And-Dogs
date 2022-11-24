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

#CNN = BaseModel("VGG1", basic_VGG, blocks=1)
#CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator)
#CNN.summarize()


#CNN = BaseModel("VGG2", basic_VGG, blocks=2)
#CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator)
#CNN.summarize()


#CNN = BaseModel("VGG3", basic_VGG, blocks=3)
#CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator)
#CNN.summarize()


#CNN = BaseModel("VGG16", transfer_VGG16)
#CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator)
#CNN.summarize()


#CNN = BaseModel("ResNet50", transfer_ResNet)
#CNN.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator)
#CNN.summarize()


ResNet = BaseModel("ResNet", transfer_ResNet, tf.keras.applications.resnet50.preprocess_input)
ResNet.run_test_harness(data_cats_dogs,["cat","dog"],DAugmentator,DAugmentatorVal)
ResNet.summarize()

###################################
