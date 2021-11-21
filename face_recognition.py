# Face Recognition

import cv2

face_cascade = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('Haarcascade/haarcascade_eye.xml') # We load the cascade for the eyes.
smile_cascade = cv2.CascadeClassifier('Haarcascade/haarcascade_smile.xml') # We load the cascade for the smile.
nose_cascade = cv2.CascadeClassifier('Haarcascade/Nose.xml') # We load the cascade for the nose

def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a blue rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 6) # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 3) # We paint a green rectangle around the eyes, but inside the referential of the face.
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(gray, 1.2, 6) #We apply the detectMultiScale method to locate the smile in the image.
        for (sx, sy, sw, sh) in smile: # For each detected smile:
            cv2.rectangle(roi_color,(sx, sy),(sx+sw, sy+sh), (0, 0, 255), 3) # We paint a red rectangle around the smile, but inside the referential of the face.
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        nose = nose_cascade.detectMultiScale(roi_gray, 1.2, 3) # We apply the detectMultiScale method to locate one or several noses in the image.
        for (nx, ny, nw, nh) in nose: # For each detected nose:
            cv2.rectangle(roi_color,(nx, ny),(nx+nw, ny+nh), (255, 0, 255), 3) # We paint a pink rectangle around the nose, but inside the referential of the face.
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    return frame # We return the image with the detector rectangles.

video_capture = cv2.VideoCapture(0) # We turn the webcam on.

while True: # We repeat infinitely (until break):
    _, frame = video_capture.read() # We get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    canvas = detect(gray, frame) # We get the output of our detect function.
    cv2.imshow('Video', canvas) # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.