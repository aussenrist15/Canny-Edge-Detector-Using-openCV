from scipy import misc
import imageio
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Defining Functions

# This function converts an RGB image to a greyscale image
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

#It is a function to apply SobelFilter(calculating gradient) on the image in a particular direction.
def SobelFilter(img, direction):
    if(direction == 'x'):
        Gx = np.array([[-1,0,+1], [-2,0,+2],  [-1,0,+1]])
        Res = ndimage.convolve(img, Gx)
        #Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
    if(direction == 'y'):
        Gy = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
        Res = ndimage.convolve(img, Gy)
        #Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)
    return Res

# This function normalize the image to make it's pixel in range (0-1)
def Normalize(img):
    #img = np.multiply(img, 255 / np.max(img))
    img = img - np.min(img)
    img = img/np.max(img)
    return img

#This function is for Non Maxima Suppression
def NonMaxSup(Gmag, Grad):
    NMS = np.zeros(Gmag.shape)
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= -22.5 and Grad[i,j] <= 22.5) or (Grad[i,j] <= -157.5 and Grad[i,j] >= 157.5)):
                if((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 22.5 and Grad[i,j] <= 67.5) or (Grad[i,j] <= -112.5 and Grad[i,j] >= -157.5)):
                if((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 67.5 and Grad[i,j] <= 112.5) or (Grad[i,j] <= -67.5 and Grad[i,j] >= -112.5)):
                if((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 112.5 and Grad[i,j] <= 157.5) or (Grad[i,j] <= -22.5 and Grad[i,j] >= -67.5)):
                if((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0

    return NMS

#This function is for Hysterisis Thresholding
def HysterisisThresholding(img):
    highThresholdRatio =0.22
    lowThresholdRatio = 0.10
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    for i in range(1,h-1):
        for j in range(1,w-1):
            if(GSup[i,j] > highThreshold):
                GSup[i,j] = 1
            elif(GSup[i,j] < lowThreshold):
                GSup[i,j] = 0
            else:
                if((GSup[i-1,j-1] > highThreshold) or
                    (GSup[i-1,j] > highThreshold) or
                    (GSup[i-1,j+1] > highThreshold) or
                    (GSup[i,j-1] > highThreshold) or
                    (GSup[i,j+1] > highThreshold) or
                    (GSup[i+1,j-1] > highThreshold) or
                    (GSup[i+1,j] > highThreshold) or
                    (GSup[i+1,j+1] > highThreshold)):
                    GSup[i,j] = 1
    return GSup




#Step 1 : Loading the Image
img = imageio.imread("Lena.png")
img = img.astype('int32')
plt.imshow(img, cmap = plt.get_cmap('gray'),vmin=0,vmax=255)
#plt.show()


#Step 2 : Applying the Gaussian FIlter to remove the unwanted noise (to smooth the image) so that only important
# edges are detected and noisy ones ignored

img_gaussian_filter = ndimage.gaussian_filter(img, sigma=1.4) # sigma is taken as 1.4
#plt.imshow(img_guassian_filter, cmap = plt.get_cmap('gray'),vmin=0,vmax=255)
#plt.show()


#Step 3 : Applying Sobel Filter (Calculating Gradient) on image

#It is to convert the image to grayscale if it is rgb
length = len(np.shape(img_gaussian_filter))
if length ==3:
    img_gaussian_filter = rgb2gray(img_gaussian_filter)


gx = SobelFilter(img_gaussian_filter, 'x')
gx = Normalize(gx)
#plt.imshow(gx, cmap = plt.get_cmap('gray'),vmin=0,vmax=255)
#plt.show()
gy = SobelFilter(img_gaussian_filter, 'y')
gy = Normalize(gy)
#plt.imshow(gy, cmap = plt.get_cmap('gray'),vmin=0,vmax=255)
#plt.show()


#Step 4 : Calculating Magnitude

Mag = np.hypot(gx,gy)
#plt.imshow(Mag, cmap = plt.get_cmap('gray'))
#plt.show()


#Step 5 : Calculating Gradients

Gradient = np.degrees(np.arctan2(gy,gx))



#Step 6 : Non Maxima Supression

WINMS = NonMaxSup(Mag,Gradient)
WINMS = Normalize(WINMS)
#plt.imshow(WINMS, cmap = plt.get_cmap('gray'))
#plt.show()

#Step 7 : Hysterisis Thresholding And Making an EdgeMap.

Final_Image = HysterisisThresholding(WINMS)
plt.imshow(Final_Image, cmap = plt.get_cmap('gray'))
plt.show()

# imgCanny = cv2.imread("Lena.png",0)
# edges = cv2.Canny(imgCanny,150,200)
# cv2.imshow("CannyOpenCV", edges)
# cv2.waitKey(0)
