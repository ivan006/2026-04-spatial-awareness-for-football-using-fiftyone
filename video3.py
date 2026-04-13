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

# 🔥 CACHE CONTOURS (faster than masks every frame)
contour_cache = {}

for img in images:
    image_id = img["id"]
    contours_list = []

    for a in anns.get(image_id, []):
        seg = a.get("segmentation")
        if isinstance(seg, dict):
            mask = mask_utils.decode(seg)
            mask = cv2.resize(mask.astype(np.uint8), (video_w, video_h))

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            contours_list.extend(contours)

    contour_cache[image_id] = contours_list

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

        for contour in contour_cache.get(image_id, []):
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    cv2.imshow("Video (Mask Outlines)", frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()