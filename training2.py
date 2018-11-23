import os
import sys
import glob
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import __version__
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import TensorBoard
from datetime import datetime

IMG_WIDTH, IMG_HEIGHT = 299, 299
NUM_EPOCHS = 50
BATCH_SIZE = 32
BATCH_SIZE_VAL = 8
FC_LAYER_SIZE = 1024
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
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_LAYER_SIZE, activation='relu')(x)
  x = Dropout(0.2)(x)
  predictions = Dense(num_classes, activation='softmax')(x)
  new_model = Model(input=model.input, output=predictions)
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
  base_model = InceptionResNetV2(weights='imagenet', include_top=False)  
  model = add_new_last_layer(base_model, num_classes)
  model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])

  model.fit_generator(
    train_generator,
    nb_epoch=NUM_EPOCHS,
    steps_per_epoch=num_train_samples*6/BATCH_SIZE,
    callbacks=[tensorboard],
    validation_data=validation_generator,
    nb_val_samples=num_val_samples,
    class_weight='auto')

  if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  
  model.save("{}/{}".format(OUTPUT_DIR,"inceptionResnet-model.h5"))
  
def main():
  train_dir = "dataset_new/training"
  val_dir = "dataset_new/testing"
  train(train_dir,val_dir)

if __name__=="__main__":
  main()
  
