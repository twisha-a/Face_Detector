import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




# Choose an image to detect faces in
# img =cv2.imread('C:/Users/Twishaa/OneDrive/Desktop/python/AI_NeuralNetworks/rdj2.jpg')
#  To capture video from webcam
webcam = cv2.VideoCapture(0)



# Iterate forever over frames
while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()


    # Must convert to gray scale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    


    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    (x, y, w, h) = face_coordinates[0]


    # Draw Rectangles around the faces 
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), ( 0, 256, 0 ), 10)
        
    cv2.imshow('AI Face Detector', frame)
    key = cv2.waitKey(1)


    ## stop if Q   key is pressed
    if key==81 or key==113:
        break


### Release the videocapture object
webcam.release()

