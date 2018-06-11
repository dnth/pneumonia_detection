import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

from keras.applications import VGG16

from keras import models
from keras import layers

from keras import optimizers

base_dir = '/home/camaro/Desktop/CODEBASE/pneumonia/chest_xray/chest_xray'

pic_normal = os.path.join(base_dir, 'train/NORMAL/IM-0151-0001.jpeg')
pic_abnormal = os.path.join(base_dir, 'train/PNEUMONIA/person1_bacteria_1.jpeg')

def load_pneumonia_model(weight_path):
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    conv_base.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['acc'])

    model.load_weights(weight_path)
    return model


def predict_pneumonia(picture_path):
    image = load_img(picture_path, target_size=(150, 150))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    return yhat

model = load_pneumonia_model('pneumonia_finetunevgg.h5')
probability_of_pneumonia = predict_pneumonia(pic_abnormal)