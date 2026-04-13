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

# sort images to match frame order
images = sorted(coco["images"], key=lambda x: x["file_name"])

cap = cv2.VideoCapture(video_path)

# get video size
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# assume Roboflow square size (adjust if needed)
img_size = 640

# compute scale (letterbox-aware)
scale = min(video_w / img_size, video_h / img_size)

new_w = int(img_size * scale)
new_h = int(img_size * scale)

pad_x = (video_w - new_w) // 2
pad_y = (video_h - new_h) // 2

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < len(images):
        image_id = images[frame_idx]["id"]

        for a in anns.get(image_id, []):
            x, y, w, h = a["bbox"]

            # scale + offset (fix alignment)
            x = int(x * scale + pad_x)
            y = int(y * scale + pad_y)
            w = int(w * scale)
            h = int(h * scale)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()