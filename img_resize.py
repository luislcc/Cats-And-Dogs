import numpy as np
import os
from torchvision.io import read_image
import torchvision.transforms as T
import threading

NUM_THREADS = 40
DIR_PATH = 'train\\'

img_paths = os.listdir('train\\')

NUM_IMGS= len(img_paths)

IMGS_PER_THREAD = NUM_IMGS//NUM_THREADS

IMG_SIZE = 200

photos, labels = [],[]


def resize2array(index):
	for i in range(index*IMGS_PER_THREAD,(index+1)*IMGS_PER_THREAD):

		path = os.path.join(DIR_PATH,img_paths[i])

		img_class = 0.0 if path.startswith('cat') else 1.0

		img = read_image(path)

		img = T.Resize(IMG_SIZE)(img)

		photos.append(img.numpy)
		labels.append(img_class)


threads = [threading.Thread(target=resize2array,args=(i,)) for i in range(NUM_THREADS)]

for i in range(NUM_THREADS):
	threads[i].start()

for i in range(NUM_THREADS):
	threads[i].join()

photos = np.asarray(photos)
labels = np.asarray(labels)

np.save('photos.npy',photos)
np.save('labels.npy',labels)
