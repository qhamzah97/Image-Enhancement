import numpy as np 
import cv2
import scipy
from scipy import misc
from scipy.signal import medfilt 

k = float(input(" How strong is Your Laplacian Note very high values are not good\n"))
print(k)
img = cv2.imread('betterplane.jpg',0)
img = scipy.signal.medfilt2d(img, 3)
cv2.imwrite("medFilt plane.png", img)
def Laplacian_Mask(img, k):
    #img = cv2.imread('bears.jpg',0)
    #img = img.astype(float)
    img = np.array(img)    
    output = np.zeros_like(img)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = img
    for x in range(img.shape[1]):     # Loop over every pixel of the image
        for y in range(img.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(k*image_padded[y:y+3,x:x+3]).sum()        
    return output

L_K = k*np.array ([[0, -1, 0 ],
               [-1, 4, -1],
               [0, -1, 0]])
impulse = ([[0, 0, 0 ],
               [0, 1, 0],
               [0, 0, 0]])

Mask_L = impulse + L_K
new_img = Laplacian_Mask(img, Mask_L)
cv2.imwrite("Laplacedplane.png", new_img)
