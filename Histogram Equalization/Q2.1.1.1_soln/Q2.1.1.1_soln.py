import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def histogram_equalize(img):
    image = np.asarray(img)
    #img = image.astype(unsigned char)
    intensity_array = np.zeros(256)
    rows, columns = image.shape[:2]
    for i in range(0, rows):
        for j in range(0, columns):
            intensity = image[i,j]
            intensity_array[intensity] = intensity_array[intensity] + 1

    #print(intensity_array)
    #print(intensity_array.shape)

    MN = 0
    for i in range(1, 256):
        MN = MN + intensity_array[i]
    #print(MN)
    
    probability_array = intensity_array/MN
    #print(probability_array)
    #print(probability_array.shape)

    CDF = 0
    CDF_array = np.zeros(256)
    for i in range(1, 256):
        CDF = CDF + probability_array[i]
        CDF_array[i] = CDF
    #print(CDF_array)
    
    final_array = np.zeros(256)
    final_array = (CDF_array * 255)
    for i in range (1,256):
        final_array[i] = math.ceil(final_array[i])
        if(final_array[i] > 255):
            final_array[i] = 255
    #print(final_array)

    new_image = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, columns):
            for value in range(0, 255):
                if (image[i,j] == value):
                    new_image[i,j] = final_array[value]
                    break
    print(new_image)
    cv2.imwrite('Sharks_New.png', new_image)

input_image = cv2.imread('sharks.png', 0)
out_image = histogram_equalize(input_image)