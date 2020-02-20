#!/usr/bin/env python
# coding: utf-8

# # Loading the necessary libraries

# In[49]:


# Import the necessary libraries
import numpy as np
import cv2 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading images
# 
# 

# In[50]:


#  Loading the image to be tested
test_image = cv2.imread('baby1.png')

# Converting to grayscale as opencv expects detector takes in input gray scale images
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Displaying grayscale image
plt.imshow(test_image_gray, cmap='gray')


# Since we know that OpenCV loads an image in BGR format so we need to convert it into RBG format to be able to display its true colours. Let us write a small function for that.

# In[51]:


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ### Loading the classifier for frontal face

# In[52]:


haar_cascade_face = cv2.CascadeClassifier('D:/Project/Intern/Face-Detection-in-Python-using-OpenCV-master/data/haarcascades/haarcascade_frontalface_alt2.xml')


# # Face detection
# 
# We shall be using the detectMultiscale module of the classifier.This function will return the co-ordinates(x and y posiiton plus the height and width) of the detected faces as Rect(x,y,w,h). 

# In[53]:


faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))      


# In[54]:


for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        


# Finally, we shall display the original image in coloured to see if the face has been detected correctly or not.

# In[55]:


#convert image to RGB and show image
plt.imshow(convertToRGB(test_image))


# ### Let us create a generalised function for the entire face detection process.

# In[64]:


def detect_faces(cascade, test_image, scaleFactor = 1.009):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()
    
    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 5)
        
    return image_copy


# ### Testing the function on new image
# 
# 
# 

# In[65]:


#loading image
test_image2 = cv2.imread('baby2.png')

#call the function to detect faces
faces = detect_faces(haar_cascade_face, test_image2)

#convert to RGB and display image
plt.imshow(convertToRGB(faces))


# ### Testing the function on a group photograph

# In[66]:


#loading image
test_image2 = cv2.imread('group.png')

#call the function to detect faces
faces = detect_faces(haar_cascade_face, test_image2)

#convert to RGB and display image
plt.imshow(convertToRGB(faces))


# ### Saving the Image

# In[67]:


cv2.imwrite('D:/Project/Intern/Face-Detection-in-Python-using-OpenCV-master/output_image.png',faces)


# In[ ]:




