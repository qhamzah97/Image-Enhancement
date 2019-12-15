import numpy as np 
import cv2
import scipy
from scipy.signal import medfilt 
import math 

pic = int(input("Pick which image to use: \n 1 - noise_additive \n 2 - noise_multiplicative \n 3 - noise_impulsive \n 4 - snowglobe \n"))
while pic < 1 or pic > 4:
        print("invalid input \n")
        pic = int(input("Pick which image to use: \n 1 - noise_additive \n 2 - noise_multiplicative \n 3 - noise_impulsive \n 4 - snowglobe \n"))

def histogram_equalize(img):
    image = np.asarray(img)
    intensity_array = np.zeros(256)
    rows, columns = image.shape[:2]
    for i in range(0, rows):
        for j in range(0, columns):
            intensity = image[i,j]
            intensity_array[intensity] = intensity_array[intensity] + 1

    MN = 0
    for i in range(1, 256):
        MN = MN + intensity_array[i]
    
    probability_array = intensity_array/MN

    CDF = 0
    CDF_array = np.zeros(256)
    for i in range(1, 256):
        CDF = CDF + probability_array[i]
        CDF_array[i] = CDF
    
    final_array = np.zeros(256)
    final_array = (CDF_array * 255)
    for i in range (1,256):
        final_array[i] = math.ceil(final_array[i])
        if(final_array[i] > 255):
            final_array[i] = 255

    new_image = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, columns):
            for value in range(0, 255):
                if (image[i,j] == value):
                    new_image[i,j] = final_array[value]
                    break
    return new_image

def Laplacian_Mask(img, k):
    img = np.array(img)    
    output = np.zeros_like(img)            # convolution output
    image_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = img
    for x in range(img.shape[1]):     # Loop over every pixel of the image
        for y in range(img.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(k*image_padded[y:y+3,x:x+3]).sum()        
    return output

def Unsharp_Mask(img, r, k):
    img = np.array(img)
    r = int(r)
    diameter = r*2+1
    count = 0
    kernal = np.zeros((int(diameter),int(diameter)))
    for i in range(0,int(diameter)):
        for j in range(0, int(diameter)):
            # assuming sigma is 1
            pi = math.pi
            kernal[i,j] = 103*((1/2*pi)*math.e**(-1*((i**2 +j**2)/(2))))
            print(kernal.astype(int))
    #convolution
    output = np.zeros_like(img) # convolution output into zero array
    for x in range(img.shape[1]):     # Loop over every pixel of the image
        for y in range(img.shape[0]):
            # element-wise multiplication of the kernel and the image    
            output[y,x] = (kernal[y:y+r , x:x+r]).sum()
    output = img + k*(img - output)
    return output

def adaptive_histogram(img, H, W):
    image = np.asarray(img)
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
    return new_image

if pic == 1:
    img = cv2.imread('noise_additive.png',0)
    out_image = histogram_equalize(img)
    cv2.imwrite("output_noise_additive.png", out_img)

elif pic == 2:
    img = cv2.imread('noise_multiplicative.png',0)
    img = scipy.signal.medfilt2d(img, 3)
    out = adaptive_histogram(img, 3, 3)
    cv2.imwrite('output_noise_multiplicative.jpg', out)

elif pic == 3:
    img = cv2.imread('noise_impulsive.png',0)
    k = float(input(" How strong is Your Laplacian Note very high values are not good\n"))
    print(k)
    img = scipy.signal.medfilt2d(img, 3)
    cv2.imwrite("medFilt_plane.png", img)
    L_K = k*np.array ([[0, -1, 0 ],
               [-1, 4, -1],
               [0, -1, 0]])
    impulse = ([[0, 0, 0 ],
               [0, 1, 0],
               [0, 0, 0]])
    Mask_L = impulse + L_K
    lpl_img = Laplacian_Mask(img, Mask_L)
    cv2.imwrite("Laplacedplane.png", lpl_img)
    out_image = histogram_equalize(lpl_img)
    cv2.imwrite("output_noise_impulsive.png", out_img)

elif pic == 4:
    img = cv2.imread('snowglobe.png',0)
    r = float(input("Give  a radius value for the blurring kernal: \n"))
    print(r)
    while r<0.0:
        print("invalid input choose another value for r")
        r = int(input("Give a radius value for :"))
        print(r)
    k = float(input(" How strong is Your Laplacian: \n"))
    print(k)
    img = scipy.signal.medfilt2d(img, 3)
    new_img = Unsharp_Mask(img,r, k)
    cv2.imwrite("UnsharpMask.png", new_img)