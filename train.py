import numpy as np 
import os
import argparse
import shutil
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator

class CNNModel(object):
	def __init__(self): 
		self.cwd = os.getcwd()
	def model(self):
		'''
		Model creation using keras
		'''
		train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.33)


		train_generator = train_datagen.flow_from_directory(directory= self.train_dir,             
		                                                    target_size=(128, 128),
		                                                    class_mode='binary',
		                                                    subset='training',
		                                                    shuffle=True,
		                                                    batch_size=32
		                                 )


		valid_generator = train_datagen.flow_from_directory(directory= self.validation_dir,
		                                                    target_size=(128, 128),
		                                                    class_mode='binary',
		                                                    shuffle = True,
		                                                    subset='validation',
		                                                    batch_size=32,
		                                                    )

		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(BatchNormalization())

		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(BatchNormalization())

		model.add(Conv2D(128, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(64, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))

		sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
		MODEL = model.fit_generator(train_generator,
		        validation_data = valid_generator,
		        validation_steps = 1,
		        epochs = 500,
		        steps_per_epoch = 500,
		        verbose = 1)
		#plot the model accuracy 
		model.save("malaria.h5")
		plt.plot(MODEL.history['acc'])
		plt.plot(MODEL.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		# plot the loss curve
		plt.plot(MODEL.history['loss'])
		plt.plot(MODEL.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()

	def data_preprocessing(self,path1,path2):

		#create parastised and uninfected folders
		self.parasited_Dir = 'parasited_Dir'
		os.makedirs(self.parasited_Dir,exist_ok=True)
		self.unifected_Dir = 'unifected_Dir'
		os.makedirs(self.unifected_Dir,exist_ok=True)

		#create train,test,validation folders
		self.train_dir = os.path.join(self.cwd, 'train')
		os.makedirs(self.train_dir,exist_ok = True)
		self.validation_dir = os.path.join(self.cwd, 'validation')
		os.makedirs(self.validation_dir,exist_ok = True)
		self.test_dir = os.path.join(self.cwd, 'test')
		os.makedirs(self.test_dir,exist_ok = True)

		#create the respective paratised and uninfected folders in the train,test and validation folders
		self.train_inf_dir = os.path.join(self.train_dir, 'parasited')
		os.makedirs(self.train_inf_dir,exist_ok = True)
		self.train_unif_dir = os.path.join(self.train_dir, 'uninfected')
		os.makedirs(self.train_unif_dir,exist_ok = True)

		self.validation_inf_dir = os.path.join(self.validation_dir, 'parasited')
		os.makedirs(self.validation_inf_dir,exist_ok = True)
		self.validation_unif_dir = os.path.join(self.validation_dir, 'uninfected')
		os.makedirs(self.validation_unif_dir,exist_ok = True)

		self.test_inf_dir = os.path.join(self.test_dir, 'parasited')
		os.makedirs(self.test_inf_dir,exist_ok = True)
		self.test_unif_dir = os.path.join(self.test_dir, 'uninfected')
		os.makedirs(self.test_unif_dir,exist_ok = True)

		#rename the folders from path1 and store in paratised folder
		i = 0		    
		for filename in os.listdir(path1): 
		    shutil.copy(path1 + filename, os.path.join(self.parasited_Dir,"parasited" + str(i) + ".jpg")) 
		    i += 1

		#rename the folders from path2 and store in uninfected folder
		j = 0
		for filename in os.listdir(path2): 
		    shutil.copy(path2 + filename, os.path.join(self.unifected_Dir,"uninfected" + str(j) + ".jpg")) 
		    j += 1       

		#copy 8000 files from paratised folder to the train/paratised folder 
		fnames = ['parasited{}.jpg'.format(i) for i in range(8000)]
		for fname in fnames:
		    shutil.copyfile(os.path.join(self.parasited_Dir, fname), os.path.join(self.train_inf_dir, fname))

		#copy 3000 files from paratised folder to the validate/paratised folder 
		fnames = ['parasited{}.jpg'.format(i) for i in range(8000, 11000)]
		for fname in fnames:
		    shutil.copyfile(os.path.join(self.parasited_Dir, fname), os.path.join(self.validation_inf_dir, fname))

		#copy renaming files from paratised folder to the test/paratised folder 
		fnames = ['parasited{}.jpg'.format(i) for i in range(11000, 13780)]
		for fname in fnames:
		    shutil.copyfile(os.path.join(self.parasited_Dir, fname), os.path.join(self.test_inf_dir, fname))

		#copy 8000 files from uninfected folder to the train/uninfected folder 
		fnames = ['uninfected{}.jpg'.format(i) for i in range(8000)]
		for fname in fnames:
		    shutil.copyfile(os.path.join(self.unifected_Dir, fname), os.path.join(self.train_unif_dir, fname))

		#copy 8000 files from uninfected folder to the validation/uninfected folder
		fnames = ['uninfected{}.jpg'.format(i) for i in range(8000, 11000)]
		for fname in fnames:
		    shutil.copyfile(os.path.join(self.unifected_Dir, fname), os.path.join(self.validation_unif_dir, fname))

		#copy 8000 files from uninfected folder to the test/uninfected folder
		fnames = ['uninfected{}.jpg'.format(i) for i in range(11000, 13780)]
		for fname in fnames:
		    shutil.copyfile(os.path.join(self.unifected_Dir, fname), os.path.join(self.test_unif_dir, fname))

		self.model()
		    
	

if __name__ == "__main__":

	PARSER = argparse.ArgumentParser(description="Present the path of Paratised and uninfected folders",
									 usage='python train.py -p1 <Paratised_Folder_path> -p2 <Uninfected_Folder_path>')
	PARSER.add_argument('--Path1', '-p1',  nargs='?',dest= "path1",
							 help='The path should contain the paratised images with.jpg/ ')	

	PARSER.add_argument('--Path2', '-p2', nargs='?', dest="path2",
							help='The path should contain the uninfected images folder path with .jpg images/')
	ARGS = PARSER.parse_args()
	Obj = CNNModel()
	Obj.data_preprocessing(ARGS.path1,ARGS.path2)
