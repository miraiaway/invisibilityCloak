import cv2
import numpy as np
import time

# Create a video capturing object and open camera for video capturing, 0 is the default camera (webcam in this case)
video_capture = cv2.VideoCapture(0)

# Sleepy time for camera to warm up, can't get up without looking at the phone
time.sleep(3)

# Create a variable called ground, instantiate to 0
background=0

# Create a for loop to capture the background frame by frame
for i in range(60):
    # Capture frame by frame
    ret, background = video_capture.read()

# Laterally (horizontally) invert the image/flip the image
background = cv2.flip(background, 1)

# While the video_capture is initialized and on....
while(video_capture.isOpened()):
    # Read each frame, place into the img variable
    ret, img = video_capture.read()
    if not ret:
        break
    img = np.flip(img,axis=1)

    # Convert from BGR to HSV color space for easier color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Range for lower red [H, S, V]
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])

    # Threshold the image to return a binary mask
    # Returns a pixel value of 1 if within range, and returns 0 if not within range
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range of red
    lower_red = np.array([170,120,70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Generate the final mask to detect the color red, these are the locations the red pixel is within the range set
    # Combine the generated masks, basically an OR operation
    mask1 = mask1 + mask2

    # Segment the detected red color
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

    # Creating an inverted mask to segment out the cloth from the frame
    mask2 = cv2.bitwise_not(mask1)

    # Segmenting the cloth out of the frame using bitwise and with the inverted mask
    res1 = cv2.bitwise_and(img,img,mask=mask2)

    # Creating image showing static background frame pixels only for the masked region
    res2 = cv2.bitwise_and(background, background, mask=mask1)

    # Generating the final output
    final_output = cv2.addWeighted(res1,1,res2,1,0)
    cv2.imshow("magic",final_output)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()



