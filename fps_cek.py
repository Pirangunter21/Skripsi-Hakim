import cv2
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage

cv2.cuda.setDevice(0)
cv2.cuda.printCudaDeviceInfo(0)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

frame_count = 0
start_time = time.perf_counter()

fps_values = []
ram_values = []
prev_time = start_time

def measure_ram_usage():
    ram_usage = memory_usage((cap.read,))
    return max(ram_usage)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    cv2.imshow("Live Streaming", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    current_time = time.perf_counter()
    elapsed_time = current_time - start_time

    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        fps_values.append(fps)

        ram_usage = measure_ram_usage()
        ram_values.append(ram_usage)

        frame_count = 0
        start_time = time.perf_counter()

cap.release()
cv2.destroyAllWindows()

# Plot kurva FPS
plt.figure(figsize=(8, 4))
plt.plot(range(len(fps_values)), fps_values)
plt.xlabel('Detik')
plt.ylabel('FPS')
plt.title('Frame per Detik')
plt.savefig('fps_graph.png')
plt.close()

# Plot kurva konsumsi RAM
plt.figure(figsize=(8, 4))
plt.plot(range(len(ram_values)), ram_values)
plt.xlabel('Detik')
plt.ylabel('Konsumsi RAM (MB)')
plt.title('Konsumsi RAM')
plt.savefig('ram_graph.png')
plt.close()


