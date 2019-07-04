from experimenting.train_model import train_classification_model
from keras_applications.inception_resnet_v2 import InceptionV3



#InceptionV3(include_top=True,
#                weights='imagenet',
#                input_tensor=None,
#                input_shape=None,
#                pooling=None,
#                classes=1000,
#                **kwargs):


batch_size = 5
nb_classes = 10
nb_epoch = 200
img_rows = 32
img_cols = 32
img_channels = 3
data_augmentation = False
train_classification_model(sample_model(img_rows,img_cols, img_channels), 'cifar10', batch_size, nb_classes, nb_epoch, img_rows, img_cols, img_channels, data_augmentation)

print('FIN')