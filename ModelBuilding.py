import time
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard

data = np.load('model_features1.npy', allow_pickle=True)

NAME="ITA MLP-{}".format(int(time.time()))
tensorboard=TensorBoard(log_dir="E:\\ITA MLP\\logs\\{}".format(NAME))

from sklearn.model_selection import train_test_split

training_data = np.asarray([i[0] for i in data])
train_labels = data[:, -1]
print("Shape of training data", training_data.shape)
print("Labels of training data", train_labels.shape)

train_data = training_data.astype('float32')
train_data = train_data / 255
from tensorflow.keras import utils as np_utils
one_hot_train_labels = np_utils.to_categorical(train_labels)

train_data1, test_data1, train_labels1, test_labels1 = train_test_split(train_data, one_hot_train_labels,
                                                                        random_state=42, test_size=0.20,shuffle=False)

import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, input_shape=(128,), activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
# opt = RMSprop(lr=0.001)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# monitor=EarlyStopping(monitor='val_loss',patience=100)
max_accuracy=0.7
for i in range(1000):
    print("Epoch no",i+1)
    history = model.fit(train_data1,train_labels1, epochs=1, batch_size=32,verbose=1,validation_data=(test_data1,test_labels1))
    if history.history['val_accuracy'][0]>max_accuracy:
        print("New best model found above")
        max_accuracy=history.history['val_accuracy'][0]
        model.save('modelupdation.h5')

model=tf.keras.models.load_model('modelupdation.h5')
[train_loss, train_accuracy] = model.evaluate(train_data1, train_labels1)
print("Evaluation result on Train Data : Loss = {}, accuracy = {}".format(train_loss, train_accuracy))
[test_loss, test_acc] = model.evaluate(test_data1, test_labels1)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


from sklearn.metrics import confusion_matrix

predictions_one_hot = model.predict(test_data1)
cm = confusion_matrix(test_labels1.argmax(axis=1), predictions_one_hot.argmax(axis=1))
print(cm)
