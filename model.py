import csv
import cv2
import os
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def loadData():
    print('loading the data...')
    lines = []
    n = 0
    fName = 'driving_log.csv'
    while os.path.isfile(fName):
        with open(fName) as f:
            content = csv.reader(f)
            fLine = True
            for line in content:
                if not fLine: lines.append(line)
                fLine = False
        n += 1
        fName = 'driving_log' + str(n) + '.csv'
    print('loaded ' + str(len(lines)) + ' images')
    return lines

def balance_data(samples, samples_needed, visualization_flag ,bins=201):
    K=bins
    del_needed = len(samples) - samples_needed
    angles = []
    for line in samples:
        angles.append(round(float(line[3]),2))
    plt.figure("Original and Modified Histograms of Image Counts")
    n, bins, patches = plt.hist(angles, bins=bins, color='#17C5E4', linewidth=0.1)
    angles = np.array(angles)
    n = np.array(n)
    idx = n.argsort()[-K:][::-1]  # find the largest K bins
    for N in reversed(range(int(n[idx[0]] - 1))):
        del_ind = [] # collect the index which will be removed from the data
        for i in range(K):
            if n[idx[i]] > N:
                ind = np.where((bins[idx[i]]<=angles) & (angles<bins[idx[i]+1]))
                ind = np.ravel(ind)
                np.random.shuffle(ind)
                if len(del_ind) + len(ind) - N <= del_needed:
                    del_ind.extend(ind[:len(ind)-N])
                else:
                    n = del_needed - len(del_ind)
                    del_ind.extend(ind[:n])
                    break
        if len(del_ind) == del_needed:
            break
    print("deleting " + str(len(del_ind)) + " images")
    # angles = np.delete(angles,del_ind)
    balanced_samples = [v for i, v in enumerate(samples) if i not in del_ind]
    balanced_angles = np.delete(angles,del_ind)
    plt.subplot(1,2,2)
    plt.hist(balanced_angles, bins=bins, color='#17C5E4', linewidth=0.1)
    plt.title('Modified Histogram', fontsize=10)
    plt.xlabel('Steering Angle', fontsize=10)
    plt.ylabel('Image Count', fontsize=10)
    if visualization_flag:
        plt.figure
        plt.subplot(1,2,1)
        n, bins, patches = plt.hist(angles, bins=bins, color='#C70039', linewidth=0.1)
        plt.title('Original Histogram', fontsize=10)
        plt.xlabel('Steering Angle', fontsize=10)
        plt.ylabel('Image Counts', fontsize=10)
        plt.show(block=False)
        plt.pause(10)
        plt.close()
    return balanced_samples

def data_augmentation(images, angles):
    augmented_images = []
    augmented_angles = []
    for image, angle in zip(images, angles):
# adjust image attributes to fit model
        image = image[60:130,:,:]
        augmented_images.append(image)
        augmented_angles.append(angle)
# flip
        flipped_image = cv2.flip(image, 1)
        flipped_angle = -1.0 * angle
        augmented_images.append(flipped_image)
        augmented_angles.append(flipped_angle)
    return augmented_images, augmented_angles

# Get full image path and file name
def get_fullName(fName):
    imagePath = "IMG/"
    fName = fName.replace("~", "")
    fName = fName[fName.rfind("\\") + 1:]
    fName = fName[fName.rfind("/") + 1:]
    fullName = imagePath + fName
    i=0
    while not os.path.isfile(fullName):
        i += 1
        cStrip = 2
        if i == 1: cStrip = 1
        imagePath = imagePath[:len(imagePath) - cStrip] + str(i) + "/"
        fullName = imagePath + fName
    return fullName

def VGG():
    model = Sequential()
    model.add(Lambda(lambda x: (x - 128) / 128, input_shape=(70, 170, 1)))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1))
    return model


def generator(samples, left_list, correction_list, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for line in batch_samples:
                angle = float(line[3])
                image = cv2.imread(get_fullName(line[0]))
                for i in range (len(left_list)):
                    correction = correction_list[i]
                    left = left_list[i]
                    right = left + 160
                    images.append(image[:,left:right,:])
                    angles.append(angle + correction)
# Augment images
            augmented_images, augmented_angles = data_augmentation(images, angles)
            X_train = np.array(augmented_images)
            clahe = cv2.createCLAHE(clipLimit=.8, tileGridSize=(4,4))
            X_train = np.array(X_train)
            X_train = np.dot(X_train[...,:3], [0.299, 0.587, 0.114])
            for i in range(X_train.shape[0]):
                X_train[i] = clahe.apply(np.uint8(X_train[i]))
                X_train[i] = cv2.equalizeHist(np.uint8(X_train[i]))
            X_train = X_train.reshape(X_train.shape + (1,))
            y_train = np.array(augmented_angles)
            yield shuffle(X_train, y_train)

# Main Program
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 512
valid_perc = .3
nSteps = 10
left_list = [0, 20, 50, 80, 110, 140, 160] # left side of each frame
correction_list = [.8, .6, .3, 0, -.3, -.6, -.8] # Angle adjustment of each frame
p_images = len(left_list) * 2
samples_needed = math.ceil(((batch_size * nSteps) / p_images) / (1 - valid_perc))
# load the csv file
samples = loadData()
# balance the data with smooth the histogram of steering angles
samples = balance_data(samples, samples_needed, visualization_flag=True)
# split data into training and validation
train_samples, validation_samples = train_test_split(samples, test_size=valid_perc)
# compile and train the model using the generator function
train_generator = generator(train_samples, left_list, correction_list,
                batch_size=batch_size)
validation_generator = generator(validation_samples,  left_list, correction_list,
                batch_size=batch_size)
# define the network model
model = VGG()
model.summary()
nEpochs = 20
adam = optimizers.Adam(lr=0.0005, beta_1=0.95)
model.compile(loss='mse', optimizer=adam)
history = model.fit_generator(train_generator, steps_per_epoch=nSteps, nb_epoch=nEpochs,
            validation_data=validation_generator, validation_steps=len(validation_samples)*p_images/batch_size)
model.save('model.h5')
print()
