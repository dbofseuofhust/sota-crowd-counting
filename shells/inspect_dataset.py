import os
import cv2

root = r'/data/deeplearning/CC/test/A'

hs, ws = [], []
for v in os.listdir(root):
    path = os.path.join(root,v)
    img = cv2.imread(path)
    if img is None:
        continue
    h,w,c = img.shape
    if h > 2048:
        hs.append(h)
    if w > 2048:
        ws.append(w)

print(len(hs),len(ws))
print(max(hs),max(ws))
print(min(hs),min(ws))