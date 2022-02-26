import cv2

# passing 0 as argument to capture video from device's camera
# we can pass the path of a video file instead to read video from that file
capture = cv2.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    cv2.imshow('Video', frame)

    # exit the window when 'd' is pressed
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()