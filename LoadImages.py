import os
import numpy as np 
from sklearn import svm 
from PIL import Image

train_images = []
test_images = []
train_labels = []
test_labels = []

f = open('labels.txt', 'r').read().split('\n')
files = os.listdir(r'')

for i in range(10000):
    pair = f[i].split(' ')
    img_object = Image.open(r''+pair[0])
    img_object.load()
    image = np.asarray(img_object).flatten()
    train_images.append(image)
    train_labels.append(pair[1])


for i in range(10000, 11000):
    pair = f[i].split(' ')
    img_object = Image.open(r''+pair[0])
    img_object.load()
    image = np.asarray(img_object).flatten()
    test_images.append(image)
    test_labels.append(pair[1])

train_images = np.asarray(train_images)/255
train_labels = np.asarray(train_labels)
test_images = np.asarray(test_images)/255
test_labels = np.asarray(test_labels)

clf = svm.SVC(kernel='linear', C=0.2187)
model = clf.fit(train_images, train_labels)
score = model.score(test_images, test_labels)

print(score)