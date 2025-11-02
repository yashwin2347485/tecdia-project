import cv2
import numpy as np
from tqdm import tqdm

INPUT_VIDEO = "jumbled_video.mp4"
OUTPUT_VIDEO = "reconstructed_final_output_video.mp4"
FPS = 30

# --- 1. Extract frames ---
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise ValueError("Could not open input video.")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = []

print(f"Extracting {frame_count} frames...")
for _ in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print("✅ Frames extracted.")

# --- 2. Preprocess frames ---
gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

# --- 3. Compute mean squared error (MSE) similarity ---
def mse(a, b):
    return np.mean((a.astype("float") - b.astype("float")) ** 2)

print("Computing frame similarities...")
n = len(gray_frames)
diff_matrix = np.zeros((n, n))
for i in tqdm(range(n)):
    for j in range(i + 1, n):
        val = mse(gray_frames[i], gray_frames[j])
        diff_matrix[i, j] = val
        diff_matrix[j, i] = val
np.fill_diagonal(diff_matrix, np.inf)

# --- 4. Choose best starting frame (least similar overall) ---
avg_diff = np.mean(diff_matrix, axis=1)
start_frame = np.argmax(avg_diff)  # least similar = probable start
print(f"Chosen starting frame: {start_frame}")

# --- 5. Greedy reconstruction ---
visited = {start_frame}
order = [start_frame]

for _ in tqdm(range(n - 1)):
    last = order[-1]
    next_idx = np.argmin(diff_matrix[last])
    while next_idx in visited:
        diff_matrix[last, next_idx] = np.inf
        next_idx = np.argmin(diff_matrix[last])
    order.append(next_idx)
    visited.add(next_idx)

# --- 6. Local smoothing (fix misplaced frames) ---
def smooth_order(order_list, gray_frames):
    improved = order_list.copy()
    for i in range(1, len(order_list) - 1):
        prev_frame = gray_frames[improved[i - 1]]
        curr_frame = gray_frames[improved[i]]
        next_frame = gray_frames[improved[i + 1]]
        if mse(prev_frame, next_frame) < mse(prev_frame, curr_frame):
            improved[i], improved[i + 1] = improved[i + 1], improved[i]
    return improved

order = smooth_order(order, gray_frames)

# --- 7. Auto-detect forward or reverse ---
def continuity(order_list):
    total = 0
    for i in range(1, len(order_list)):
        total += mse(gray_frames[order_list[i-1]], gray_frames[order_list[i]])
    return total

forward_score = continuity(order)
reverse_score = continuity(order[::-1])
if reverse_score < forward_score:
    order = order[::-1]
    print("🔁 Reversed order for smoother continuity.")
else:
    print("➡ Forward order chosen.")

# --- 8. Write reconstructed video ---
print("Saving reconstructed video...")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (frame_width, frame_height))
for idx in order:
    out.write(frames[idx])
out.release()

print(f"🎉 Done! Reconstructed video saved as {OUTPUT_VIDEO}")
