#%%
# build a cnn that can classify men and women from images

from keras import layers
from keras import models
from keras.regularizers import l2
import numpy as np
import os
import random
import cv2

# resize the images 

s = 256
folder_list = os.listdir('balanced')

with open('sex_dictionary', 'r') as f:
    sex_dict = eval(f.read())

for k in sex_dict.copy():
    if k not in folder_list:
        del sex_dict[k]

images = []
labels = []

for folder in folder_list:
    for im in os.listdir(os.path.join('balanced', folder)):
        img = cv2.imread(os.path.join('balanced', folder, im))
        img = cv2.resize(img, (s, s))
        img = img / 255
        images.append(img)

        if sex_dict[folder] == 0:
            labels.append(0)
        
        else:
            labels.append(1)

print(len(images))
print(len(labels))

# shuffle the images and the labels together
data = list(zip(images, labels))

# Shuffle the combined list
random.shuffle(data)

# Split the shuffled data into training and testing sets,
split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

# Unzip the training and testing sets back into separate lists
train_images, train_labels = zip(*train_data)
test_images, test_labels = zip(*test_data)

print(len(train_images))
print(len(train_labels))
print(len(test_images))
print(len(test_labels))
#%%
# build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(s, s, 3),
                                                kernel_regularizer=l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
#%%
# fit the model
history = model.fit(np.array(train_images), np.array(train_labels), epochs=10,
                    batch_size=32, validation_split=0.2)
#%%
print(model.evaluate(np.array(test_images), np.array(test_labels)))