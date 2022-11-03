from models import BaseModel,basic_VGG,transfer_VGG16
from models_utility import DataAugmentator
import tensorflow as tf
from organizer import DirectoryOrganizer,ImageQuery



GPU
print(tf.config.list_physical_devices('GPU'))


#################################
### Meta Work ###

#data_cats_dogs = "CatsDogs"
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



##################################
### Testes ###

VGG1 = BaseModel("VGG1",basic_VGG(1))
VGG1.run_test_harness(data_cats_dogs)
VGG1.summarize()
VGG1.final_test_harness(final_cats_dogs)

VGG2 = BaseModel("VGG2",basic_VGG(2))
VGG2.run_test_harness(data_cats_dogs)
VGG2.summarize()
VGG2.final_test_harness(final_cats_dogs)

VGG3 = BaseModel("VGG3",basic_VGG(3))
VGG3.run_test_harness(data_cats_dogs)
VGG3.summarize()
VGG3.final_test_harness(final_cats_dogs)

VGG16 = BaseModel("VGG16",transfer_VGG16())
VGG16.run_test_harness(data_cats_dogs)
VGG16.summarize()
VGG16.final_test_harness(final_cats_dogs)

da = DataAugmentator()
VGG16 = BaseModel("VGG16_DA",transfer_VGG16())
VGG16.run_test_harness(data_cats_dogs,da.values())
VGG16.summarize()
VGG16.final_test_harness(final_cats_dogs)

###################################




# make sure it runs on a GPU
# make models for +2 elements dataSet



