import csv
import cv2
import os
import numpy as np
import time
import math
import matplotlib.pyplot as plt

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

def balance_data(samples, samples_needed, bins=201):
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
    return balanced_samples

def data_augmentation(images, angles):
    augmented_images = []
    augmented_angles = []
    for image, angle in zip(images, angles):
# adjust image attributes to fit model
        image = image[60:130,:,:]
        image = cv2.resize(image, (32, 32), interpolation = cv2.INTER_AREA)
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


def makeImages(records, left_list, correction_list):
    num_records = len(records)
    np.random.shuffle(records)
    images = []
    angles = []
    for i in range(num_records):
        angle = float(records[i][3])
        image = cv2.imread(get_fullName(records[i][0]))
        for j in range (len(left_list)):
            correction = correction_list[j]
            left = left_list[j]
            right = left + 160
            images.append(image[:,left:right,:])
            angles.append(round(angle + correction, 2))
# Augment images
        augmented_images, augmented_angles = data_augmentation(images, angles)
    X = np.array(augmented_images)
    X = np.array(X)
    X = np.dot(X[...,:3], [0.299, 0.587, 0.114])
    clahe = cv2.createCLAHE(clipLimit=.8, tileGridSize=(4,4))
    for j in range(X.shape[0]):
        X[j] = clahe.apply(np.uint8(X[j]))
        X[j] = cv2.equalizeHist(np.uint8(X[j]))
    X = X.reshape(X.shape + (1,))
    y = np.array(augmented_angles)
    return(X, y)

def plot_images(fname, X, y = None, indices = None, cols = 5):
    if indices is None:
        indices = range(len(X))
    rows = np.ceil(len(indices) / cols).astype('uint32')
    plt.figure(figsize = (15, rows * 2))
    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    for i, index in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        if y is not None:
            plt.title("Angle " + str(y[index]), fontsize=10)
        plt.xticks([])
        plt.yticks([])
        if X[index].shape[2] == 1:
            plt.imshow(X[index][:,:,0], cmap='gray')
        else:
            plt.imshow(colorCLAHE(X[index]), cmap='gray')
    plt.savefig(fname)
    plt.show()

def plot_random_samples(X, y, fname, num = 5, from_index = 0):
    samples = np.random.randint(from_index, len(X), size=num)
    plot_images(fname, X, y, samples)
    return samples

def getFileName(name):
    return "media/" + name.replace(" ", "_") + ".png"

# Main Program
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
left_list = [0, 20, 50, 80, 110, 140, 160] # left side of each frame
correction_list = [.8, .6, .3, 0, -.3, -.6, -.8] # Angle adjustment of each frame
batch_size = 512
valid_perc = .3
nSteps = 10
left_list = [0, 20, 50, 80, 110, 140, 160] # left side of each frame
correction_list = [.8, .6, .3, 0, -.3, -.6, -.8] # Angle adjustment of each frame
p_images = len(left_list) * 2
samples_needed = math.ceil(((batch_size * nSteps) / p_images) / (1 - valid_perc))
# load the csv file
records = loadData()
# balance the data with smooth the histogram of steering angles
records = balance_data(records, samples_needed)
# images
X, y = makeImages(records, left_list, correction_list)
name = "Random Images"
fname = getFileName(name)
_ = plot_random_samples(X, y, fname, 50)
