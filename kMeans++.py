from math import sqrt
from random import randint
import numpy as np
from numpy.core.fromnumeric import shape
import sys
from basicAgent import colorDistance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common import convertToGrayscale, getImagePixels


def plot3D(pixels):
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    red = []
    green = []
    blue = []
    for r in pixels:
        for c in r:
            #print(c)
            red.append(c[0])
            green.append(c[1])
            blue.append(c[2])
    ax.scatter(red, green, blue, marker='.')
    ax.scatter([208], [231], [249], marker='s')
    ax.scatter([3], [0], [0], marker='s')
    ax.scatter([255], [243], [14], marker='s')
    ax.scatter([255], [21], [30], marker='s')
    ax.scatter([43], [180], [255], marker='s')
    plt.show()
    #print(pixels.shape)

def kMeansPP1(pixels, k, distance=colorDistance):
    '''
        Algo:
        1) Pick random first center, add to centers list
        2) Find farthest point from this, add this to centers list
        3) for each point that is not a center, find distances to each of the centers
            a) keep track of the average of the distances from the centers
        4) the point with the highest distance average will be the next center to append to the list

    '''
    centers = []
    leftPixels = pixels[:, :int(pixels.shape[1] / 2)]
    firstC = leftPixels[randint(0, leftPixels.shape[0] - 1), randint(0, leftPixels.shape[1] - 1)]
    centers.append(firstC)

    maxDistance = 0
    nextC = []
    for r in leftPixels:
        for c in r:
            dist = distance(c, centers[0])
            if dist > maxDistance:
                maxDistance = dist
                nextC = c.copy()
    centers.append(nextC)
    print(centers)


    
    newCenter = []
    while len(centers) < k:
        distance = 0
        for r in leftPixels: # gotta figure out how to not have duplicate centers now
            for c in r:
                total = 0
                for center in centers:
                    d = colorDistance(c, center)
                    total += d
                total = total / len(centers)
                if total > distance:
                    distance = total
                    newCenter = c.copy()
        centers.append(newCenter)
        print(centers)


    

    return centers


def kMeansPP(pixels, k, distance=colorDistance):
    leftPixels = pixels[:, :int(pixels.shape[1] / 2)]
    centers = []
    firstC = leftPixels[randint(0, leftPixels.shape[0] - 1), randint(0, leftPixels.shape[1] - 1)]
    centers.append(firstC)
    print('first center:', firstC)


    for c_id in range(k-1):
        dist = []
        for r in range(leftPixels.shape[0]):
            for c in range(leftPixels.shape[1]):
                point = leftPixels[r,c]
                d = sys.maxsize
                #print('point:', point)

                for h in range(len(centers)):
                    #print('h:', h)
                    #print('center?: ', centers[h])
                    temp_dist = distance(point, centers[h]) 
                    #print('dist:', temp_dist)
                    d = min(d, temp_dist)  
                #print('d: ', d)
                dist.append(d)
                #print(dist.shape)

            dist = np.array(dist)
            print('dist_shape:', dist.shape)
            #print('distances: ', dist)
            print(np.argmax(dist))
            nextC = leftPixels[np.argmax(dist), :, :]
            #print('next center:', nextC)
            centers.append(nextC)
        #print('next center:', nextC)
        dist = []
    return centers
    
    
    
    
    # r = c = 0
    # while r < pixels.shape[0]:
    #     while c < pixels.shape[1]:
    #         pixelRGB = pixels[r,c]
            

    # return

if __name__ == "__main__":
    originalPixels = getImagePixels("training", "fuji.jpg")
    plot3D(originalPixels)
    #kMeansPP1(originalPixels, 5)