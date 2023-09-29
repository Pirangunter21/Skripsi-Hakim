import cv2
import numpy as np
import tensorflow as tf
import psutil
from tkinter import *
from PIL import ImageTk

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# model = tf.saved_model.load('/home/sentjet/Documents/kode/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8/saved_model')
model = tf.saved_model.load('/home/sentjet/Documents/kode/lite-20230523T015954Z-001/lite/fine_tuned_model/saved_model')
# model = tf.saved_model.load('/home/sentjet/Documents/kode/lite40-20230523T085109Z-001/lite40/fine_tuned_model/saved_model')
# model = tf.saved_model.load('/home/sentjet/Documents/kode/mnet_5-20230523T090138Z-001/mnet_5/fine_tuned_model/saved_model')

fps_values = []
ram_hardware = psutil.virtual_memory().total / 1024 / 1024

root = Tk()
root.title("Object Detection")

fps_label = Label(root, text="FPS Kamera: 0.00", font=("Arial", 12))
fps_label.pack()

ram_label = Label(root, text="Konsumsi RAM: 0.00 MB", font=("Arial", 12))
ram_label.pack()

frame_label = Label(root, text="Frame Kamera Deteksi", font=("Arial", 12))
frame_label.pack()

frame_canvas = Canvas(root, width=640, height=480)
frame_canvas.pack()

def update_ui():
    fps = 0.0 if len(fps_values) == 0 else np.mean(fps_values)
    fps_label.config(text=f"FPS Kamera: {fps:.2f}")
    ram_label.config(text=f"Konsumsi RAM: {ram_hardware:.2f} MB")
    root.after(1000, update_ui)

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        input_tensor = np.expand_dims(frame, axis=0)
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

        frame_image = ImageTk.PhotoImage(image=Image.fromarray(frame))
        frame_canvas.create_image(0, 0, anchor=NW, image=frame_image)
        frame_canvas.image = frame_image

    frame_canvas.after(1, update_frame)

update_ui()
update_frame()

root.mainloop()



