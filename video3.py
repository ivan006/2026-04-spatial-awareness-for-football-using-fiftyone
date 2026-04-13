import cv2
import json
import numpy as np
from pycocotools import mask as mask_utils

video_path = "dataset/petal_20260413_065046.mp4"
coco_path = "dataset/test/train/_annotations.coco.json"

with open(coco_path) as f:
    coco = json.load(f)

# map image_id -> annotations
anns = {}
for a in coco["annotations"]:
    anns.setdefault(a["image_id"], []).append(a)

images = sorted(coco["images"], key=lambda x: x["file_name"])

cap = cv2.VideoCapture(video_path)

video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 🔥 CACHE MASKS (huge speed boost)
mask_cache = {}

for img in images:
    image_id = img["id"]
    masks = []

    for a in anns.get(image_id, []):
        seg = a.get("segmentation")
        if isinstance(seg, dict):
            mask = mask_utils.decode(seg)
            mask = cv2.resize(mask.astype(np.uint8), (video_w, video_h))
            masks.append(mask)

    mask_cache[image_id] = masks

frame_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        # 🔁 LOOP VIDEO
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        continue

    if frame_idx < len(images):
        image_id = images[frame_idx]["id"]

        overlay = frame.copy()

        for mask in mask_cache.get(image_id, []):
            color = np.zeros_like(frame, dtype=np.uint8)
            color[:, :] = (0, 255, 0)
            overlay[mask == 1] = color[mask == 1]

        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.imshow("Video (Mask Overlay)", frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()