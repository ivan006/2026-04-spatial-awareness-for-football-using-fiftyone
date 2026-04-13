import cv2
import json

video_path = "dataset/petal_20260413_065046.mp4"  # your original video
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

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < len(images):
        image_id = images[frame_idx]["id"]

        for a in anns.get(image_id, []):
            x, y, w, h = map(int, a["bbox"])
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()