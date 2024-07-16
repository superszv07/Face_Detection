import cv2    // import openCv Library          
 // put the path of the file for detection in which there is some inbuild algorithm
face_cap = cv2.CascadeClassifier("C:/Users/Admin/OneDrive/Documents/pythonpro/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)      // open camera for capture image and the argument (0) means capture image at runtime
while True:                          // here we use while loop for contionuously capturing the image until we want close it.
    ret, vid = video_cap.read()      // read the image capture by video_cap
    col = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY) // ust to change the color into gray to better  face details
    faces = face_cap.detectMultiScale(       // used for face surface details as data
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces: // use for diplay rectange on detected face
        cv2.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 2) // vid->show rectangle, (x,y)->height and width, (x+w,y+h)-> to display rectangle width and height, (0,255,0)-> color ,2-> width of outer line
    cv2.imshow("WebCam", vid)        //to display frame of image which is read by vid
    if cv2.waitKey(10) == ord("q"):  // close the frame after pressing "q"
        break
video_cap.release()                  // release capturing of image

// 1 ,4 ,5 ,6,17,,18,19,20 for open the camera for capture image
//3,7,8 for getting face details as data
//15,16 for display rectangle on  detected face
