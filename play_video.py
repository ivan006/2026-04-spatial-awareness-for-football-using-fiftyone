import cv2
import json
import os

images_dir = "dataset/test/train"
coco_path = "dataset/test/train/_annotations.coco.json"

with open(coco_path) as f:
    coco = json.load(f)

# map image_id -> annotations
anns = {}
for a in coco["annotations"]:
    anns.setdefault(a["image_id"], []).append(a)

# sort images by file name (important)
images = sorted(coco["images"], key=lambda x: x["file_name"])

for img in images:
    path = os.path.join(images_dir, img["file_name"])
    frame = cv2.imread(path)

    for a in anns.get(img["id"], []):
        x, y, w, h = map(int, a["bbox"])
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Video", frame)
    if cv2.waitKey(50) & 0xFF == 27:  # ESC to quit
        break

cv2.destroyAllWindows()