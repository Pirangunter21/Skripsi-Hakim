import cv2
import tensorflow as tf
import tensorflow_hub as hub
import time
import os
import matplotlib.pyplot as plt

model_path = 'https://tfhub.dev/tensorflow/efficientdet/lite4/1'  
cap = cv2.VideoCapture(0)
model = tf.saved_model.load(model_path)
input_shape = [1, 640, 640, 3]
score_threshold = 0.5

fps_values = []
detection_times = []
detection_speeds = []

start_time = time.time()
frame_count = 0

output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)

save_image = False  # Penanda apakah gambar harus disimpan

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (640, 640))
    input_tensor = tf.convert_to_tensor(resized_frame)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    start_detection_time = time.time()

    output_dict = model(input_tensor)

    end_detection_time = time.time()
    detection_time = end_detection_time - start_detection_time
    fps = 1 / detection_time if detection_time != 0 else 0

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
            class_name = str(class_id)
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

# Plot the FPS and Detection Speed curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(detection_times, fps_values)
plt.title('FPS over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('FPS')

plt.subplot(1, 2, 2)
plt.plot(detection_times, detection_speeds)
plt.title('Detection Speed over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Detection Speed (objects per second)')

plt.tight_layout()
plt.show()

# Save the FPS, detection time, and detection speed data to a text file
output_file = open("detection_data.txt", "w")
output_file.write("FPS Values:\n")
output_file.write(str(fps_values))
output_file.write("\n\nDetection Times:\n")
output_file.write(str(detection_times))
output_file.write("\n\nDetection Speeds:\n")
output_file.write(str(detection_speeds))
output_file.close()

