import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
import os
import csv
from memory_profiler import memory_usage
from jtop import jtop
import psutil

# Inisialisasi jtop
jetson = jtop()

# Menampilkan informasi perangkat GPU yang tersedia
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Physical GPU is available")
tf.debugging.set_log_device_placement(True)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.grab()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

model = tf.saved_model.load('/home/sentjet/Documents/kode/custom/hasil128/saved_model')

input_shape = [1, 800, 600, 3]
score_threshold = 0.5

category_index = {
    1: 'bearbrand_normal', 2: 'bearbrand_rusak'
}

frame_count = 0
start_time = time.perf_counter()

fps_values = []
ram_values = []
gpu_values = []
cpu_values = []
prev_time = start_time

output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)
save_image = False

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
record = False
out = None

def measure_ram_usage():
    ram_usage = memory_usage((cap.read,))
    return max(ram_usage)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    resized_frame = cv2.resize(frame, (800, 600))
    input_tensor = tf.convert_to_tensor(resized_frame)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    # Memproses input_tensor menggunakan GPU
    with tf.device('/GPU:0'):
        output_dict = model(input_tensor)

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
            cv2.putText(frame, f"{class_name}: {score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            num_detections += 1

    current_time = time.perf_counter()
    elapsed_time = current_time - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        fps_values.append(fps)
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_count = 0
        start_time = time.perf_counter()
        ram_usage = measure_ram_usage()
        ram_values.append(ram_usage)

        gpu_usage = jetson.gpu.get("usage", 0)
        cpu_usage = psutil.cpu_percent()
        gpu_values.append(gpu_usage)
        cpu_values.append(cpu_usage)

    cv2.imshow("Live Streaming", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord('f'):
        save_image = True
    elif key == ord('r'):
        record = not record
        if record:
            if out is None:
                video_name = f"captured_images/video_{time.strftime('%Y%m%d%H%M%S')}.mp4"
                out = cv2.VideoWriter(video_name, fourcc, 20.0, (800, 600))
                print(f"Recording started: {video_name}")
        else:
            if out is not None:
                out.release()
                out = None
                print("Recording stopped.")

    if save_image:
        image_name = f"captured_images/image_{time.strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(image_name, frame)
        print(f"Image saved: {image_name}")
        save_image = False

cap.release()
cv2.destroyAllWindows()

# Menulis data ke file CSV
output_file = open("detection_data.csv", "w", newline='')
csv_writer = csv.writer(output_file)
csv_writer.writerow(['Time count', 'Frame', 'Ram Usage', 'CPU Usage', 'GPU Usage'])
for i in range(len(fps_values)):
    csv_writer.writerow([elapsed_time, fps_values[i], ram_values[i], cpu_values[i], gpu_values[i]])
output_file.close()

plt.figure(figsize=(8, 4))
plt.plot(range(len(fps_values)), fps_values)
plt.xlabel('Detik')
plt.ylabel('FPS')
plt.title('Frame per Detik')
plt.savefig('fps_graph.png')
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(range(len(ram_values)), ram_values)
plt.xlabel('Detik')
plt.ylabel('Konsumsi RAM (%)')
plt.title('Konsumsi RAM')
plt.savefig('ram_usage_graph.png')
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(range(len(cpu_values)), cpu_values)
plt.xlabel('Detik')
plt.ylabel('Konsumsi CPU (%)')
plt.title('Konsumsi CPU')
plt.savefig('cpu_usage_graph.png')
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(range(len(gpu_values)), gpu_values)
plt.xlabel('Detik')
plt.ylabel('Konsumsi GPU (%)')
plt.title('Konsumsi GPU')
plt.savefig('gpu_usage_graph.png')
plt.close()


