import cv2
import json

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

print("Video:", video_w, "x", video_h)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < len(images):
        img_info = images[frame_idx]
        image_id = img_info["id"]

        # 🔥 USE REAL IMAGE SIZE FROM COCO
        img_w = img_info["width"]
        img_h = img_info["height"]

        scale_x = video_w / img_w
        scale_y = video_h / img_h

        for a in anns.get(image_id, []):
            x, y, w, h = a["bbox"]

            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()