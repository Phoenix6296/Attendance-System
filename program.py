import face_recognition
import cv2
import numpy as np
import os
import csv
from datetime import datetime

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Loop and load the image and face encoding for each person in the known_people folder
known_face_encodings = []
known_face_names = []
for file in os.listdir("Images"):
    # Hidden File in Mac
    if file == '.DS_Store':
        continue
    face_name = face_recognition.load_image_file("Images/" + file)
    face_encoding = face_recognition.face_encodings(face_name)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(file.split(".")[0])

students = known_face_names.copy()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
s = True


# Initialize the date and time
now = datetime.now()
current_date = now.strftime("%H:%M:%S")

# Initialize the csv file
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"
            face_distance = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()
