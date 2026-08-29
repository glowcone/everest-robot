"""Run the detector over sampled frames and build a contact sheet to eyeball."""

import glob
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/stephn/Documents/docs/code/everest-hack")
from carabiner.detect import NotFound, detect, draw

files = sorted(glob.glob("out/look/*.png"))
tiles, ok = [], 0
for f in files:
    im = cv2.imread(f)
    name = f.split("/")[-1].replace(".png", "")
    try:
        t = detect(im)
        v = draw(im, t)
        ok += 1
        label = f"{name} a={t.area:.0f} th={t.spine_angle:+.0f}"
        col = (0, 255, 0)
    except NotFound as e:
        v = im.copy()
        label = f"{name} MISS {e}"[:44]
        col = (0, 0, 255)
    cv2.rectangle(v, (0, 0), (v.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(v, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    tiles.append(cv2.resize(v, (320, 240)))

print(f"detected {ok}/{len(files)}")
cols = 5
rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
if len(rows[-1].shape) and rows[-1].shape[1] != rows[0].shape[1]:
    pad = np.zeros((240, rows[0].shape[1] - rows[-1].shape[1], 3), np.uint8)
    rows[-1] = np.hstack([rows[-1], pad])
cv2.imwrite("out/sheet.png", np.vstack(rows))
print("wrote out/sheet.png")
