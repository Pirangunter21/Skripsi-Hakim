import cv2
import numpy as np
import tensorflow as tf
import time
import os
import psutil
import csv
import matplotlib.pyplot as plt
import pynvml

print(tf.config.list_physical_devices('GPU'))
tf.debugging.set_log_device_placement(True)
cv2.cuda.setDevice(0)
cv2.cuda.printCudaDeviceInfo(0)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
model = tf.saved_model.load('/home/sentjet/Documents/kode/custom/ssdv2.1/saved_model')
# model = tf.saved_model.load('/home/sentjet/Documents/kode/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
input_shape = [1, 640, 480, 3]
score_threshold = 0.5

category_index = {
     1: 'bb_normal', 2: 'bb_rusak' 
}

fps_values = []
detection_times = []
detection_speeds = []
gpu_power_values = []
cpu_power_values = []
cpu_freq_values = []
gpu_freq_values = []
power_draw_values = []
current_values = []
voltage_values = []

start_time = time.time()
frame_count = 0

ram_usage = []
start_ram = psutil.virtual_memory().used / (1024 * 1024)

# NVIDIA GPU power measurement initialization
pynvml.nvmlInit()
device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

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

    # GPU power measurement
    gpu_power = pynvml.nvmlDeviceGetPowerUsage(device_handle) / 1000.0  # in watts
    gpu_power_values.append(gpu_power)

    # CPU power measurement
    cpu_power = psutil.cpu_percent() / 100.0 * psutil.cpu_freq().max / 1000.0  # in watts
    cpu_power_values.append(cpu_power)

    # CPU frequency measurement
    cpu_freq = psutil.cpu_freq().current / 1000.0  # in GHz
    cpu_freq_values.append(cpu_freq)

    # GPU frequency measurement
    gpu_freq = pynvml.nvmlDeviceGetClockInfo(device_handle, pynvml.NVML_CLOCK_GRAPHICS) / 1000.0  # in GHz
    gpu_freq_values.append(gpu_freq)

    # Power draw measurement
    power_draw = psutil.sensors_power().power_draw  # in watts
    power_draw_values.append(power_draw)

    # Current measurement
    current = psutil.sensors_battery().current  # in Amperes
    current_values.append(current)

    # Voltage measurement
    voltage = psutil.sensors_battery().voltage  # in Volts
    voltage_values.append(voltage)

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
csv_writer.writerow(['FPS', 'Detection Time', 'Detection Speed', 'GPU Power', 'CPU Power', 'CPU Frequency',
                     'GPU Frequency', 'Power Draw', 'Current', 'Voltage'])
for i in range(len(fps_values)):
    csv_writer.writerow([fps_values[i], detection_times[i], detection_speeds[i], gpu_power_values[i],
                         cpu_power_values[i], cpu_freq_values[i], gpu_freq_values[i], power_draw_values[i],
                         current_values[i], voltage_values[i]])
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

plt.figure(figsize=(12, 6))
plt.plot(detection_times, gpu_power_values)
plt.title('GPU Power Consumption over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('GPU Power Consumption (W)')
plt.savefig('gpu_power_curve.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(detection_times, cpu_power_values)
plt.title('CPU Power Consumption over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('CPU Power Consumption (W)')
plt.savefig('cpu_power_curve.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(detection_times, cpu_freq_values)
plt.title('CPU Frequency over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('CPU Frequency (GHz)')
plt.savefig('cpu_freq_curve.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(detection_times, gpu_freq_values)
plt.title('GPU Frequency over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('GPU Frequency (GHz)')
plt.savefig('gpu_freq_curve.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(detection_times, power_draw_values)
plt.title('Power Draw over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Power Draw (W)')
plt.savefig('power_draw_curve.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(detection_times, current_values)
plt.title('Current over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Current (A)')
plt.savefig('current_curve.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(detection_times, voltage_values)
plt.title('Voltage over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (V)')
plt.savefig('voltage_curve.png')
plt.close()


