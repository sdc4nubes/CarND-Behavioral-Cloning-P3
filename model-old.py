import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

def plot_images(datatype, names, x, y, cols):

    datasize = len(x)
    indices = range(len(x))
    if datasize>50:
        indices=random.sample(indices,50)

    plt.close()
    rows = np.ceil(len(indices) / cols).astype('uint32')
    plt.figure(figsize = (15, rows * 2))
    plt.suptitle('Random ' + datatype + ' Samples', y=.95 , fontsize=24)
    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    for i, index in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        y[index] = round(y[index],4)
        plt.title("Class " + str(y[index]), fontsize=10, fontweight='bold')
        plt.text(16, 35, names[index][-7:],
                verticalalignment='top',
                horizontalalignment='center',
                color='black', fontsize=10)
        plt.xticks([])
        plt.yticks([])
        if x[index].shape[2] == 1:
            plt.imshow(x[index][:,:,0], cmap='gray')
        else:
            plt.imshow(x[index], cmap='gray')
    plt.show()

def plot_samples(datatype, x, y, names, num = 10, from_index = 0):
    plot_images(datatype, y, x, names, cols=num)
    return

# Get data path
def path():
    path = os.pardir + "/Self Driving Car Beta Simulator Data/"
    return path

# Get data boundaries
def getMinMax(category_width):
    half_width = category_width/2.
    minRange = -1 - half_width
    maxRange = 1 + half_width
    return minRange, maxRange

# Greyscale, contrast, crop, extend and normalize the data
def augment_data(images, names, angles, model_type, category_width, plot=False):
    augmented_images = []
    augmented_angles = []
    augmented_names = []
    minRange, maxRange = getMinMax(category_width)
    correction = .5
#Greyscale the data
    #images = np.dot(images[...,:3], [0.299, 0.587, 0.114])
    #clahe = cv2.createCLAHE()
    for image, fName, angle in zip(images, names, angles):
#Apply Contrast limited Adaptive Histogram Equalization
        #image = clahe.apply(np.uint8(image))
        #image = cv2.GaussianBlur(image, (3,3), 0)
#Crop and augment the data
        icorrection = correction
        for i in range(5):
            x1 = (i * 37) + 1
            x2 = x1 + 170
            image1 = image[60:130, x1:x2]
            image1 = cv2.resize(image1, (32,32), interpolation = cv2.INTER_AREA)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
            angle1 = angle + icorrection
            if model_type == 'cross':
                angle1 = round(angle1, 1)
            if angle1 <= maxRange and angle1 >= minRange:
                augmented_images.append(image1)
                augmented_angles.append(angle1)
                augmented_names.append(fName)
            icorrection -= correction/2

#Normalize the data
    augmented_images = np.array(augmented_images)
    #augmented_images = ((augmented_images - 128) / 128.).astype(np.float32)
#Change color dimension from RGB to Grey
    #augmented_images = augmented_images.reshape(augmented_images.shape + (1,))
    if plot:
        plot_data('augmented histogram', augmented_angles, category_width)

    return augmented_images, augmented_names, augmented_angles

# Get full image path and file name
def get_fullName(fName):
    imagePath = path() + "IMG/"
    flip = False
    if fName[0:] == "~":
        flip = True
    fName = fName.replace("~", "")
    fName = fName[fName.rfind("\\")+1:]
    fName = fName[fName.rfind("/")+1:]
    fullName = imagePath + fName
    i=0
    while not os.path.isfile(fullName):
        i+=1
        cStrip = 2
        if i == 1:
            cStrip = 1
        imagePath = imagePath[:len(imagePath)-cStrip] + str(i) + "/"
        fullName = imagePath + fName
    return fullName, flip

# Preprocess all images
def preprocess_images(names, angles, model_type, category_width, plot=False):
    images=[]
    fNames=[]
    for i in range(len(names)):
        fullName, flip = get_fullName(names[i])
        image = cv2.imread(fullName)
        if flip:
            cv2.flip(image, 0)
        images.append(image)
        fName = fullName[fullName.rfind("/")+1:]
        fNames.append(fName)
    images = np.array(images)
    a_images, a_names, a_angles = \
        augment_data(images, fNames, angles, model_type, category_width, plot)
    if plot:
        plot_samples("Augmented", a_images, a_names , a_angles)
    return a_images, a_names, a_angles

### The following functions were adapted from:
### https://github.com/JunshengFu/driving-behavioral-cloning

# Load and flip csv data
def load_and_flip_CSVs(model_type, category_width):
    print('loading the data...')
    steering_angles = []
    image_names = []
    index = []
    n=0
    fName = path() + 'driving_log.csv'
    while os.path.isfile(fName):
        print(fName)
        with open(fName) as f:
            content = csv.reader(f)
            for line in content:
                if 'center' in line[0]:
                    fName, _ = get_fullName(line[0])
                    image_names.append(fName)
                    angle = float(line[3])
                    if model_type == 'cross':
                        angle = round(angle,1)
                    steering_angles.append(angle)
        n+=1
        fName = path() + 'driving_log' + str(n) + '.csv'

    print('flipping the data...')
    nRecs = len(steering_angles)
    for i in range(nRecs):
        image_names.append("~" + image_names[i])
        angle = -steering_angles[i]
        steering_angles.append(angle)

    return image_names, steering_angles

# Create histogram plots
def plot_data(title, data, category_width, show=True):
    nbins = int(1/category_width*2+1)
    minRange, maxRange = getMinMax(category_width)
    plt.figure
    plt.style.use("seaborn-dark")
    plt.hist(data, range=(minRange,maxRange),
        bins=nbins, linewidth=category_width)
    fsize = 10
    plt.title(title, fontsize=fsize)
    plt.xlabel('steering angle', fontsize=fsize)
    plt.ylabel('counts', fontsize=fsize)
    if show:
        plt.show()

#Remove names when the number of occurences of a given angle exceeds N
def balance_data(names, angles, category_width, plot=False ,N=60):
    fsize = 10
    nbins = getNumOfCats(category_width)
    n, bins, patches = plt.hist(angles, bins=nbins)
    angles=np.array(angles)
    n = np.array(n)
    print('number of categories = ' + str(n.shape[0]))

    K=nbins
    idx = n.argsort()[-K:][::-1]   # find the largest K bins
    del_ind = [] # collect the indices which will be removed from the data
    for i in range(K):
        if n[idx[i]] > N:
            ind = np.where((bins[idx[i]]<=angles) & (angles<bins[idx[i]+1]))
            ind = np.ravel(ind)
            np.random.shuffle(ind)
            del_ind.extend(ind[:len(ind)-N])

    print("deleting " + str(len(del_ind)) + " images")

    balanced_names = [v for i, v in enumerate(names) if i not in del_ind]
    balanced_angles = np.delete(angles, del_ind)

    if plot:
        plt.subplot(1,2,2)
        plot_data('modified histogram', balanced_angles, category_width, False)
        plt.subplot(1,2,1)
        plot_data('original histogram', angles, category_width)
    else:
        plt.close()

    return balanced_names, balanced_angles

def getNumOfCats(category_width):
    return int(1/category_width*2+2)

# Generate data in subsets
def generator(samples, model_type, category_width, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = [i[0] for i in batch_samples]
            angles = [i[1] for i in batch_samples]
            X_train = np.array(images)
            y_train = np.array(angles)
            if model_type == 'cross':
                nClasses = getNumOfCats(category_width)
                y_train = to_categorical(y_train, num_classes=nClasses)
            yield shuffle(X_train, y_train)

def crossentropy_model(category_width):
    model = Sequential()
    model.add(Convolution2D(32,(3,3),input_shape=(70,170,1),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(128,(3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(256,(3,3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(getNumOfCats(category_width), activation='softmax'))

    return model

def mse_model(category_width):
    model = Sequential()
    model.add(Convolution2D(32,(3,3),input_shape=(70,170,1),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(128,(3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(256,(3,3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(0.5))
    model.add(Dense(20))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model

# Main Program
model_type = 'mse'
category_width=.1
# load the csv files
image_names, steering_angles = load_and_flip_CSVs(model_type, category_width)
print('number of images = ' + str(len(image_names)))
# Balance the number of data images in each category
b_image_names, b_steering_angles = balance_data(image_names, steering_angles,
    category_width, True)
print('number of balanced images = ' + str(len(b_image_names)))

# Preprocess the data (used for testing only)
a_images, a_image_names, a_steering_angles = \
    preprocess_images(b_image_names, b_steering_angles, model_type, category_width, True)
print('number of augmented images = ' + str(len(a_steering_angles)))
a_samples = list(zip(a_images, a_steering_angles))

# split data into training and validation
train_samples, validation_samples = train_test_split(a_samples, test_size=0.3)
print('number of training samples = ' + str(len(train_samples)))
print('number of validation samples = ' + str(len(validation_samples)))

# compile and train the model using the generator function
batch_size=256
train_generator = generator(train_samples, model_type, category_width, batch_size)
validation_generator = generator(validation_samples, model_type, category_width, batch_size)

# define the network model
if model_type == 'cross':
    model = crossentropy_model(category_width)
else:
    model = mse_model(category_width)
model.summary()

nbEpoch = 4
if model_type == 'cross':
    model.compile(loss='categorical_crossentropy', optimizer='adam')
else:
    model.compile(loss='mse', optimizer='adam')

history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, nb_epoch=nbEpoch, validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size)

model.save('model.h5')
