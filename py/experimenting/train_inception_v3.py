from train_model import train_classification_model
import keras

batch_size = 5
nb_classes = 10
nb_epoch = 200
img_rows = 32
img_cols = 32
img_channels = 3
data_augmentation = False
reshapeTo = (75, 75)
#https://keras.io/applications/#inceptionv3

model = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=None, input_shape=(75, 75, 3), pooling=None, classes=nb_classes)
train_classification_model(model, 'cifar10', batch_size, nb_classes, nb_epoch, img_rows, img_cols, img_channels, data_augmentation = True, reshapeTo = reshapeTo,  featurewise_center=True, samplewise_center=True,  featurewise_std_normalization=True,  samplewise_std_normalization=True)

print('FIN')