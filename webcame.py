import cv2
face_cap = cv2.CascadeClassifier("C:/Users/Admin/OneDrive/Documents/pythonpro/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)
while True:
    ret, vid = video_cap.read()
    col = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("WebCam", vid)

    if cv2.waitKey(10) == ord("q"):
        break
video_cap.release()
