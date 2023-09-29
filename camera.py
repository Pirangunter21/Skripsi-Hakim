import cv2
import numpy as np
import tensorflow as tf
import psutil
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
model = tf.saved_model.load('/home/sentjet/Documents/kode/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
fps_values = []
ram_hardware = psutil.virtual_memory().total / 1024 / 1024
while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (640, 480))
    input_tensor = np.expand_dims(resized_frame, axis=0)
    input_tensor = tf.cast(input_tensor, tf.uint8)
    fps_start_time = cv2.getTickCount()
    output_dict = model(input_tensor)
    fps_end_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (fps_end_time - fps_start_time)
    fps_values.append(fps)
    boxes = output_dict['detection_boxes'][0].numpy()
    scores = output_dict['detection_scores'][0].numpy()
    classes = output_dict['detection_classes'][0].numpy().astype(np.int32)
    score_threshold = 0.5
    for i in range(len(scores)):
        if scores[i] > score_threshold:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            with open('mscoco_label_map.pbtxt', 'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'id' in line and str(classes[i]) in line:
                    name = line.split(':')[1].strip().replace("'", "")
                    break
            else:
                name = 'unknown'
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    fps_text = f"FPS Kamera: {fps:.2f}  FPS Real-Time: {np.mean(fps_values):.2f}"
    cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (208, 240, 192), 2)
    ram_text = f"Konsumsi RAM: {ram_hardware:.2f} MB"
    cv2.putText(frame, ram_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (208, 240, 192), 2)
    cv2.imshow('Object Detection', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





