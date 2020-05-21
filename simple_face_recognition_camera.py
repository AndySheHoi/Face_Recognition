import cv2
import face_recognition

# =============================================================================
# read the image from the local path
# =============================================================================

# Please load your own image
me = cv2.imread('Images/Me.jpg')


# =============================================================================
# Process all images
# =============================================================================

# face encoding for all known face
me_face_encoding = face_recognition.face_encodings(me)[0]

# Face Encoding List
known_face_encodings = [me_face_encoding]

# Name List
known_face_names = ['Me']


# =============================================================================
# Recognize all faces in the video camera and show the result
# =============================================================================

# Use your video camera
vc = cv2.VideoCapture(0)

# Process the image returned by the video camera
while True:
    ret, img = vc.read()
    
    # if you did not open your camera, it will print 'No image captured'
    if ret == False:
        print('No image captured')
        break

    # Detect all face locations in the video camera    
    locations = face_recognition.face_locations(img)
    
    # face encoding for the video
    unknow_face_encodings = face_recognition.face_encodings(img, locations)
    
    
    # Compare and recognize each detected face, if it matches, mark the corresponding name.
    # If the match is unsuccessful, it will be marked as Nobody.
    # location: [top, right, bottom, left]

    for (top, right, bottom, left), face_encoding in zip(locations, unknow_face_encodings):
        
        matchings = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = 'NoBody'
        for matching, known_name in zip(matchings, known_face_names):
            if matching:
                name = known_name
                break

        # Draw a frame on the detected face and place the name in the upper left corner of the frame          
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
        cv2.putText(img, name, (left, top-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    
    # Show the result
    cv2.imshow('Video', img)
   
    # Press any button on the keyboard to exit
    if cv2.waitKey(1) != -1:
        vc.release()
        cv2.destroyAllWindows()
        break

