import cv2
import numpy as np
import tensorflow as tf
import time
import os
import psutil
import csv
import matplotlib.pyplot as plt
print(tf.config.list_physical_devices('GPU'))
tf.debugging.set_log_device_placement(True)

# CUDA initialization
cv2.cuda.setDevice(0)
cv2.cuda.printCudaDeviceInfo(0)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
# model = tf.saved_model.load('/home/sentjet/Documents/kode/custom/ssdv2.2/hasil/saved_model')
model = tf.saved_model.load('/home/sentjet/Documents/kode/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model')
# model = tf.saved_model.load('/home/sentjet/Documents/kode/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
input_shape = [1, 640, 480, 3]
score_threshold = 0.5

category_index = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

fps_values = []
detection_times = []
detection_speeds = []

start_time = time.time()
frame_count = 0

ram_usage = []
start_ram = psutil.virtual_memory().used / (1024 * 1024)                        

output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)
save_image = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (640, 480))
    input_tensor = tf.convert_to_tensor(resized_frame)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    start_detection_time = time.time()

    output_dict = model(input_tensor)

    end_detection_time = time.time()
    detection_time = end_detection_time - start_detection_time
    fps = 1 / detection_time if detection_time != 0 else 0
    ram = psutil.virtual_memory().used / (1024 * 1024)
    ram_usage.append(ram - start_ram)

    boxes = output_dict['detection_boxes'][0].numpy()
    scores = output_dict['detection_scores'][0].numpy()
    classes = output_dict['detection_classes'][0].numpy()

    for i in range(len(scores)):
        if scores[i] > score_threshold:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            class_id = int(classes[i])
            class_name = category_index[class_id]
            confidence = scores[i]  # Confidence value
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            if save_image:
                # Save the captured image
                image_name = f"{class_name}_{int(time.time())}.jpg"
                image_path = os.path.join(output_dir, image_name)
                cv2.imwrite(image_path, frame.copy())
                save_image = False

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Detection Time: {detection_time:.5f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.putText(frame, f"RAM Usage: {ram_usage[-1]:.2f} MB", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Object Detection', frame)
    fps_values.append(fps)
    detection_times.append(time.time() - start_time)
    detection_speeds.append(len(scores) / detection_time)
    frame_count += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('f'):
        save_image = True

cap.release()
cv2.destroyAllWindows()

output_file = open("detection_data.csv", "w", newline='')
csv_writer = csv.writer(output_file)
csv_writer.writerow(['FPS', 'Detection Time', 'Detection Speed'])
for i in range(len(fps_values)):
    csv_writer.writerow([fps_values[i], detection_times[i], detection_speeds[i]])
output_file.close()

plt.figure(figsize=(12, 6))
plt.plot(detection_times, fps_values)
plt.title('FPS over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('FPS')
plt.savefig('fps_curve.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(detection_times, detection_speeds)
plt.title('Detection Speed over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Detection Speed (objects per second)')
plt.savefig('detection_speed_curve.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(detection_times, ram_usage)
plt.title('RAM Usage over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('RAM Usage (MB)')
plt.savefig('ram_usage_curve.png')
plt.close()


