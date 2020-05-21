import cv2
import face_recognition


# =============================================================================
# read the image from the local path
# =============================================================================

# load images to set up a 'face database'
trump = cv2.imread('Images/Trump.jpg')
obama = cv2.imread('Images/Obama.jpg')

# load test image
img = cv2.imread('Images/barack-obama-and-donald-trump.jpg')


# =============================================================================
# Process all images
# =============================================================================

# detect all face locations in the image
locations = face_recognition.face_locations(img)

# face encoding for the test image
unknow_face_encodings = face_recognition.face_encodings(img, locations)

# face encoding for all known face
obama_face_encoding = face_recognition.face_encodings(obama)[0]
trump_face_encoding = face_recognition.face_encodings(trump)[0]

# Face Encoding List
known_face_encodings = [trump_face_encoding, obama_face_encoding]

# Name List
known_face_names = ['Trump', 'Obama']


# =============================================================================
# Recognize all faces in the image
# =============================================================================

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
    

# =============================================================================
# Show the result
# =============================================================================

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
