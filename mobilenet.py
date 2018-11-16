import os
import sys
import glob
#import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import __version__
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout,Conv2D,Activation,Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import TensorBoard
from datetime import datetime
from PIL import Image, ImageFile


IMG_WIDTH, IMG_HEIGHT = 224, 224
NUM_EPOCHS = 25
BATCH_SIZE = 32
BATCH_SIZE_VAL = 8
NUM_LAYERS_TO_FREEZE = 249
OUTPUT_DIR = "output"
LOG_DIR = "logs"

def get_nb_files(directory):
  if not os.path.exists(directory):
    return 0
  count = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      count += len(glob.glob(os.path.join(r, dr + "/*")))
  return count

def add_new_last_layer(model, num_classes):
  x = model.output
  x = Reshape((1, 1, 1024), name='reshape_1')(x)
  x = Dropout(0.001, name='dropout')(x)
  x = Conv2D(num_classes, (1, 1),padding='same',name='conv_preds')(x)
  x = Activation('softmax', name='act_softmax')(x)
  x = Reshape((num_classes,), name='reshape_2')(x)
  new_model = Model(input=model.input, output=x)
  return new_model

def train(train_dir,val_dir):
  num_train_samples = get_nb_files(train_dir)
  num_classes = len(glob.glob(train_dir + "/*"))
  num_val_samples = get_nb_files(val_dir)
  
  #tensorboard setting
  if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
  tensorboard = TensorBoard(log_dir="{}/{}".format(LOG_DIR,datetime.now().strftime('%Y%m%d-%H%M%S')))

  # data prep
  train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

  test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

  train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
  )

  validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE_VAL,
  )

  #transfer learning
  base_model = MobileNet(weights='imagenet', include_top=False,input_shape=(224,224,3),pooling='avg')
  model = add_new_last_layer(base_model, num_classes)
  model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])
  
  model.fit_generator(
    train_generator,
    nb_epoch=NUM_EPOCHS,
    steps_per_epoch=num_train_samples*6/BATCH_SIZE,
    callbacks=[tensorboard],
    validation_data=validation_generator,
    nb_val_samples=num_val_samples*6,
    class_weight='auto')

  # model.save('Transfer_learning.h5')

  # history_transfer_learning = model.fit_generator(
  #   train_generator,
  #   nb_epoch=NUM_EPOCHS,
  #   samples_per_epoch=num_train_samples,
  #   validation_data=validation_generator,
  #   nb_val_samples=num_val_samples,
  #   class_weight='auto')
	
  #show_plot(history_transfer_learning)
  
  # fine-tuning
  # for layer in model.layers[:NUM_LAYERS_TO_FREEZE]:
  #    layer.trainable = False
  # for layer in model.layers[NUM_LAYERS_TO_FREEZE:]:
  #    layer.trainable = True
  # #model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
  # model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])

  # model.fit_generator(
  #   train_generator,
  #   steps_per_epoch=num_train_samples*6/BATCH_SIZE,
  #   nb_epoch=NUM_EPOCHS,
  #   callbacks=[tensorboard],
  #   validation_data=validation_generator,
  #   nb_val_samples=num_val_samples,
  #   class_weight='auto')

  # history_fine_tuning = model.fit_generator(
  #   train_generator,
  #   samples_per_epoch=num_train_samples,
  #   nb_epoch=NUM_EPOCHS,
  #   validation_data=validation_generator,
  #   nb_val_samples=num_val_samples,
  #   class_weight='auto')
  if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  
  model.save("{}/{}".format(OUTPUT_DIR,"mobilenet.h5"))
  
def main():
  ImageFile.LOAD_TRUNCATED_IMAGES = True
  train_dir = "dataset_newest/Train"
  val_dir = "dataset_newest/Validation"
  train(train_dir,val_dir)

if __name__=="__main__":
  #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  # config = tf.ConfigProto()
  # config.gpu_options.per_process_gpu_memory_fraction = 0.7
  # session = tf.Session(config=config)
  main()
  
