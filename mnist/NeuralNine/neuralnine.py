from tkinter import image_names
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1 )  
x_test = tf.keras.utils.normalize(x_test,axis=1 )

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

model.fit(x_train,y_train,epochs=50)

model.save("handwritten.model")

model = tf.keras.models.load_model('handwritten.model')

loss,accuracy=model.evaluate(x_test,y_test)
print(loss)
print(accuracy)

image_number = 1
while os.path.isfile(f"digits/{image_number}.png"):
    try:
        img = cv2.imread(f"digits/{image_number}.png")[:,:,0] #grayscaling
        img = np.invert(np.array([img])) #Inverting the image as the original images are white and contain black background.check digital sreeni tut 20 for futher processing understanding
        prediction = model.predict(img)
        print(f"This digit is probably a %d"%(np.argmax(prediction)))
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        image_number+=1