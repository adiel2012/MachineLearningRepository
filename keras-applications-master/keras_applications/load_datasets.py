import tensorflow as tf
import pathlib
import random

#import numpy as np
from functools import reduce
import os, sys
from PIL import Image
#from __future__ import absolute_import, division, print_function






def loadimg(img_path):
    photo = Image.open(img_path) #your image
    #photo = photo.convert('RGB')
    result = []
    width = photo.size[0] #define W and H
    height = photo.size[1]
    for y in range(0, height): #each pixel has coordinates
        row = []
        for x in range(0, width):
            col = []
            RGB = photo.getpixel((x,y))
            R,G,B = RGB  #now you can use the RGB value
            col =  [[R,G,B]]
            row = row + col
        result = result + [row]
    return result

def loadDS(folder_url):
    data_root = folder_url
    data_root = pathlib.Path(data_root)
    #print(data_root)

    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    #print(image_count)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    #print(label_names)
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    label_to_index
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

    result = list( map(loadimg, all_image_paths) )

    return (result, all_image_labels)

#BATCH_SIZE = 4
dataset_name = 'cifar10'
abs_path = os.path.abspath('..\\datasets\\'+dataset_name+'\\train\\')
(x, y) = loadDS(abs_path)
#X = np.array(x)
#Y = np.array(y)
#print(X.shape)
#print(Y.shape)
