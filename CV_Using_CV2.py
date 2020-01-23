import numpy as np
import matplotlib.pyplot as plt
import cv2
%matplotlib inline

path ="/home/azad/Data_Science_Courses/01_DataScience-20190528T065230Z-001/01_DataScience/Datasets/sammy.jpg"

img = cv2.imread(path)
img

plt.imshow(img)

# The image has been correctly loaded by openCV as a numpy array,
# but the color of each pixel has been sorted as BGR.

# swap those channels.
# This can be done by using openCV conversion functions cv2.cvtColor()
# or by working directly with the numpy array.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # this swaps the channels
# this is converting it to bgr to rgb
#cv2.imshow('RGB',img_rgb)
#cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
#cv2.namedwindow('RGB',cv2.Window_Autosize)
#cv2.waitKey(0)
#cv2.destroyWindow('RGB') 

plt.imshow(img_rgb)

img_gray = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
plt.imshow(img_gray)
print(img_gray.shape)

# why it is not in BW, coz of the order of channels
img_gray = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
plt.imshow(img_gray,cmap='gray')
print(img_gray.shape)

img_gray = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
plt.imshow(img_gray,cmap='magma')
print(img_gray.shape)

# ntc it has only 1 channel
print(img_gray.min()) # min pxl val
print(img_gray.max()) # max pxl val# ## Resize Images, Img manipulation
plt.imshow(img_rgb) # this is the orgnl img
img_rgb.shape

# width, height, color channels
img =cv2.resize(img_rgb,(1000,400)) # resize it, this will squeeze it
plt.imshow(img) # ntc the X-axis & Y-axis
plt.show()
img =cv2.resize(img_rgb,(1300,275)) # resize it
plt.imshow(img)
plt.show()

# ### Resize By ratio
w_ratio = 0.8
h_ratio = 0.2
new_img =cv2.resize(img_rgb,(0,0),img,w_ratio,h_ratio) # 80% smaller
plt.imshow(new_img)
plt.show()
w_ratio = 0.5 # 50% of orgnl width
h_ratio = 0.5 # 50% of orgnl ht
new_img =cv2.resize(img_rgb,(0,0),img,w_ratio,h_ratio) # 50% smaller
plt.imshow(new_img)
plt.show()
plt.imshow(img_rgb)
plt.show()
print(new_img.shape)
print(img_rgb.shape)
#get_ipython().magic('matplotlib qt')
plt.imshow(new_img)
plt.show()




# ### Flipping Images
#get_ipython().magic('matplotlib inline')# Along central x axis
new_img = cv2.flip(new_img,0) # 0 is hor axis
plt.imshow(new_img)
plt.show()
# Along central y axis
new_img = cv2.flip(new_img,1) # 1 is vert axis
plt.imshow(new_img)
plt.show()
# Along both axis
new_img = cv2.flip(new_img,-1)
plt.imshow(new_img)
plt.show()# # Saving Image Files
type(new_img)

