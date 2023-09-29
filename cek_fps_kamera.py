import cv2

cv2.cuda.setDevice(0)
cv2.cuda.printCudaDeviceInfo(0)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
while True:
    ret, frame = cap.read()       
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    print("fps:", fps)     
    font = cv2.FONT_HERSHEY_SIMPLEX  
    cv2.putText(frame, str(fps), (50, 50), font, 1, (0, 0, 255), 2) 
    cv2.imshow("Live Streaming", frame)     
    key = cv2.waitKey(1)  
    if key == ord("q"):  
        break
 
cap.release()
cv2.destroyAllWindows()

