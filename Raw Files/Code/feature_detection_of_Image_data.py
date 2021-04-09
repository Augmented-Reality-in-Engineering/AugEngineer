import cv2
import numpy as np
from imutils import paths        #Imutils are a series of convenience function
#It will make basic image processing functions such as translation,rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV.

ImagePath = list(paths.list_images('dc'))

# feature detection in a dc image data set

orb = cv2.ORB_create(nfeatures=1500)   #initialize the instance for orb

feature_detection = []
for image in ImagePath:
    images = cv2.imread(image, cv2.IMREAD_GRAYSCALE)     # read all images in grayscale
    resize = cv2.resize(images, (300, 250), interpolation=cv2.INTER_AREA)  # resize those images
    keypoints, descriptors = orb.detectAndCompute(resize, None)  # finding and initializing the keypoints of images
    img = cv2.drawKeypoints(resize, keypoints, images, (0, 255, 0))  # draw those keypoints
    feature_detection.append(img)


feature_detection = np.array(feature_detection)
for pics in feature_detection:
    cv2.imshow('DC_Pics', pics)             # show the images with keypoints
    if cv2.waitKey(0) & 0xFF == ord('s'):   # when we press s so the window will be terminated
        break
cv2.destroyAllWindows()