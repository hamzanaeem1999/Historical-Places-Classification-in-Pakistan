import cv2
import os
import numpy as np
import random

DataDir = r"C:\Users\Hamza\OneDrive\Desktop\All Images1\Training Data"
Categories = ["Badshahi Masjid","Faisal Masjid","Minare Pakistan","Quaid's Tomb","ShahiQila(Lahore Fort)"]
descriptors = []
class_num1 = []
sift = cv2.xfeatures2d.SIFT_create(40)

training_data = []


def create_training_data():
    for category in Categories:
        path = os.path.join(DataDir, category)
        class_num = Categories.index(category)
        IMG_SIZE = (124, 124)
        for img in os.listdir(path):
            try:
                global new_array, desImage
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_arr, IMG_SIZE)
                keyImage, desImage = sift.detectAndCompute(new_array, None)
                # np.savetxt('des/' + img[:-3] + '.csv', desImage, delimiter='\t', fmt='%1.4e')
                feat = np.sum(desImage, axis=0)  # columns wise adding
                training_data.append([feat, class_num])
                # print("# kps: {}, filename:{}, descriptors: {}".format(len(keyImage), img, desImage.shape))
            except Exception as e:
                pass


create_training_data()
random.shuffle(training_data)
np.save('desc_feat.npy', np.asarray(training_data))
print('Shape', np.asarray(training_data).shape)
