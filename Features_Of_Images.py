import cv2
import numpy as np
import tensorflow as tf

Categories = ["Badshahi Masjid", "Minare Pakistan", "ShahiQila(Lahore Fort)"]

sift = cv2.xfeatures2d.SIFT_create()


def prepare(filepath):
    IMG_SIZE = 124, 124
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, IMG_SIZE)
    keyImage, desImage = sift.detectAndCompute(new_array, None)
    desImage=(desImage/255).astype('float32')
    feat = np.sum(desImage, axis=0)
    return feat.reshape(-1, 128)


model = tf.keras.models.load_model("Superclass_model2.h5")

prediction = model.predict([prepare(r'C:\Users\Hamza\OneDrive\Desktop\MinarePakistan\m6.jpeg')])
print("Probability of predictions are: ",prediction)
highest=np.argmax(prediction)
print("Highest position : ", highest)
# print("Specified class is: ",Categories[int(prediction[0][0])])
