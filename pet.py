import os
import cv2
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import natsort

np.set_printoptions(suppress=True)



image_size = (224, 224)
batch_size = 32

train_ds = keras.preprocessing.image_dataset_from_directory(
    "data2",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    "data2",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Entry block
    #x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    #(image_array.astype(np.float32) / 127.0) - 1
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    activation = "softmax"
    units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)



def visualize(train_ds):
	plt.figure(figsize=(10, 10))
	for images, labels in train_ds.take(18):
	    for i in range(9):
	        ax = plt.subplot(3, 3, i + 1)
	        #images[i] = (images[i].numpy().astype(np.float32) / 127.0) - 1
	        plt.imshow(images.numpy().astype("uint8"))
	        plt.title(int(labels[i]))
	        plt.axis("off")
	plt.show()



'''
#training
model = make_model(input_shape=image_size + (3,), num_classes=3)
#print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=40, validation_data=val_ds, verbose=2)
model.save('model/pet_model_m_1.h5')
print("\nmodel is saved")

'''



def test_folder(f_name):
	filenames = [f_name + x for x in os.listdir(f_name)]
	filenames = natsort.natsorted(filenames)
	data = np.ndarray(shape=(len(filenames), 224, 224, 3), dtype=np.float32)
	#print(filenames)
	i=0
	for file in filenames:
		images = Image.open(file)
		image_array = np.asarray(images)

		image_array = (image_array.astype(np.float32) / 127.0) - 1

		data[i] = image_array
		i+=1

	pred = model.predict(data)
	os.system('cls')
	for ix in range(len(pred)):
		
		if pred[ix].argmax(axis=-1)==0:
			label='YES: clear'
		elif pred[ix].argmax(axis=-1)==1:
			label='NO: color PET'
		else:
			label='NO: contains garbage'	
		print('\nResult',ix+1,':', label)
		
		#print('\nResult: ', 'clear' if pred[ix].argmax(axis=-1)==0 else 'not_clear')
		print('Confidence: ', 'clear-%.1f  color-%.1f  garbage-%.1f' % (pred[ix,0]*100, pred[ix,1]*100, pred[ix,2]*100))



def test_single():
	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	image = Image.open('test/21.jpg')
	#size = (224, 224)
	#image = ImageOps.fit(image, size, Image.ANTIALIAS)
	image_array = np.asarray(image)
	image.show()

	image_array = (image_array.astype(np.float32) / 127.0) - 1

	image_array.show()
	data[0] = image_array
	pred = model.predict(data)
	if pred.argmax(axis=-1)==0:
		label='YES: clear'
	elif pred.argmax(axis=-1)==1:
		label='NO: color PET'
	else:
		label='NO: contains garbage'
	#print('\nResult: ', 'clear' if pred.argmax(axis=-1)==0 else 'not_clear')
	print('\nResult: ', label)
	print('Confidence: ', 'clear-%.1f  color-%.1f  garbage-%.1f' % (pred[0,0]*100, pred[0,1]*100, pred[0,2]*100))




def test_webcam():
	video = cv2.VideoCapture(0)
	#print('width: ', video.get(3))
	#print('height: ', video.get(4))
	while True:
	        _, frame = video.read()
	        #frame = cv2.flip(frame, 1)
	        #frame = cv2.resize(frame, (224, 224))
	        #frame = frame[128:352,208:432]
	        im = Image.fromarray(frame, 'RGB')
	        im = ImageOps.fit(im, (224,224), Image.ANTIALIAS)
	        #im = im.resize((224,224))
	        img_array = np.array(im)
	        
	        img_array = (img_array.astype(np.float32) / 127.0) - 1

	        img_array = np.expand_dims(img_array, axis=0)
	        pred = model.predict(img_array)
	        rgb=(0,0,255)
	        #label='not_detected'

	        if pred.argmax(axis=-1)==0:
	        	label='YES: clear'
	        	rgb=(0,255,0)
	        elif pred.argmax(axis=-1)==2:
	        	label='NO: contains garbage'
	        	rgb=(0,0,255)
	        else:
	        	label='NO: color PET'
	        	rgb= (255,0,255)
	        
	        
	        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, rgb, 2)
	        cv2.imshow("Capturing", frame)
	        key=cv2.waitKey(1)
	        if key == ord('q'):
	        	break
	video.release()
	cv2.destroyAllWindows()




#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

model = load_model('model/pet_model_5.h5')

test_webcam()
#test_folder('test/')
