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

cv2.cuda.setDevice(0)
cv2.cuda.printCudaDeviceInfo(0)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
model = tf.saved_model.load('/home/sentjet/Documents/kode/custom/hasil3/saved_model')

input_shape = [1, 800, 600, 3]
score_threshold = 0.5

category_index = {
    1: 'kaleng_indomilk_cacat', 2: 'kaleng_indomilk_normal', 3: 'kaleng_milo_cacat', 4: 'kaleng_milo_normal'
}

fps_values = []
ram_usage = []
start_time = time.perf_counter()

output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)
save_image = False

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
record = False
out = None

frame_count = 0
total_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    resized_frame = cv2.resize(frame, (800, 600))
    input_tensor = tf.convert_to_tensor(resized_frame)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    start_time_frame = time.perf_counter()
    output_dict = model(input_tensor)
    end_time_frame = time.perf_counter()
    detection_time = end_time_frame - start_time_frame
    total_time += detection_time
    fps = frame_count / total_time
    ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    ram_usage.append(ram)

    boxes = output_dict['detection_boxes'][0].numpy()
    scores = output_dict['detection_scores'][0].numpy()
    classes = output_dict['detection_classes'][0].numpy()

    num_detections = 0

    for i in range(len(scores)):
        if scores[i] > score_threshold:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])

            class_id = int(classes[i])
            class_name = category_index[class_id]
            score = scores[i]

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            num_detections += 1

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"RAM Usage: {ram_usage[-1]:.2f} MB", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)
    fps_values.append(fps)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('f'):
        save_image = True
    elif key == ord('r'):
        record = not record
        if record:
            print("Recording started.")
        else:
            if out is not None:
                out.release()
                out = None
            print("Recording stopped.")

cap.release()
cv2.destroyAllWindows()

time_values = np.arange(len(fps_values)) / fps

ram_usage = np.array(ram_usage) - ram_usage[0]

plt.figure(figsize=(12, 6))
plt.plot(time_values, fps_values)
plt.title('FPS over Time')
plt.xlabel('Time (s)')
plt.ylabel('FPS')
plt.savefig('fps_curve.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(time_values, ram_usage)
plt.title('RAM Usage over Time')
plt.xlabel('Time (s)')
plt.ylabel('RAM Usage (MB)')
plt.savefig('ram_usage_curve.png')
plt.show()


