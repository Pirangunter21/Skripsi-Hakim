import cv2

# Set up GStreamer pipeline
def gstreamer_pipeline(capture_width=3280, capture_height=2464, display_width=820, display_height=616, framerate=21, flip_method=2):
    return ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)%d, height=(int)%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv flip-method=%d ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! appsink' % (capture_width, capture_height, framerate, flip_method, display_width, display_height))

# Create VideoCapture object
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Failed to open camera!")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# Set MJPG as the FourCC codec
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Get FPS
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS Kamera:", fps)

# Initialize frame count
frame_count = 0

# Start capturing frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Kamera', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release VideoCapture object
cap.release()

# Close all windows
cv2.destroyAllWindows()

# Calculate FPS
fps_calculated = frame_count / fps
print("FPS yang dihitung:", fps_calculated)
