import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def adaptive_histogram(img, H, W):
    image = np.asarray(img)
    #img = image.astype(unsigned char)
    rows, columns = image.shape[:2]
    
    cutlength_x = math.floor(W/2)
    cutlength_y = math.floor(H/2)

    range_w = range(-math.floor(W/2), math.ceil(W/2))
    range_h = range(-math.floor(H/2), math.ceil(H/2))

    intensity_array = np.zeros(256)
    MN = H * W
    CDF = 0
    count = 0
    CDF_array = np.zeros(256)
    final_array = np.zeros(256)
    new_image = np.zeros(img.shape)
    
    for i in range (cutlength_x, rows-cutlength_x):
        for j in range (cutlength_y, columns-cutlength_y):
            for x in range_w:
                for y in range_h:
                    intensity = image[x,y]
                    intensity_array[intensity] = intensity_array[intensity] + 1

                    probability_array = intensity_array/MN

                    CDF = 0
                    CDF_array = np.zeros(256)
                    for l in range(1, 256):
                        CDF = CDF + probability_array[l]
                        CDF_array[l] = CDF
                    
                    final_array = np.zeros(256)
                    final_array = (CDF_array * 255)
                    for h in range (1,256):
                        final_array[h] = math.ceil(final_array[h])
                        if(final_array[h] > 255):
                            final_array[h] = 255

                    for value in range(0, 255):
                        if (image[i+x,j+y] == value):
                            new_image[i,j] = final_array[value]
                            break
    print(1)
    #print(new_image)
    #cv2.imwrite('Bears_New.jpg', new_image)
    cv2.imwrite('Mask_New.jpg', new_image)

input_image = cv2.imread('bears.jpg', 0)
mask = np.array([[10,20,30,40,50,60,70,80,90],
                 [15,25,35,45,55,65,75,85,95],
                 [20,30,40,50,60,70,80,90,100],
                 [25,35,45,55,65,75,85,95,105],
                 [30,40,50,60,70,80,90,100,110],
                 [35,45,55,65,75,85,95,105,115],
                 [40,50,60,70,80,90,100,110,120],
                 [45,55,65,75,85,95,105,115,125],
                 [50,60,70,80,90,100,110,120,130]])
cv2.imwrite('Mask.jpg', mask)
#out = adaptive_histogram(input_image, 3, 3)
out = adaptive_histogram(mask, 3, 3)
