import numpy as np 
import cv2
import scipy
from scipy.signal import medfilt 
import math 

r = float(input("Give  a radius value for the blurring kernal: \n"))
print(r)
while r<0.0:
    print("invalid input choose another value for r")
    r = int(input("Give a radius value for :"))
    print(r)
k = float(input(" How strong is Your Laplacian: \n"))
print(k)
img = cv2.imread('betterplane.jpg',0)
img = scipy.signal.medfilt2d(img, 3)

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
       # Add zero padding to the input image
    #image_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))   
    #image_padded[-1:1, -1:1] = img
    for x in range(img.shape[1]):     # Loop over every pixel of the image
        for y in range(img.shape[0]):
            # element-wise multiplication of the kernel and the image    
            output[y,x] = (kernal[y:y+r , x:x+r]).sum()
    output = img + k*(img - output)
    return output


new_img = Unsharp_Mask(img,r, k)
cv2.imwrite("Laplacedplane.png", new_img)