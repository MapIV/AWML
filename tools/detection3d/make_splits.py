#!/usr/bin/env python3
import os, json, random

ROOT = "./data/nuscenes"
SPLIT_DIR = os.path.join(ROOT, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

scene_file = os.path.join(ROOT, "v1.0-trainval", "scene.json")
with open(scene_file, "r") as f:
    scenes = json.load(f)

scene_names = [s["name"] for s in scenes]
print(f"Totalscene: {len(scene_names)}")

# random split 80/20
random.seed(42) 
random.shuffle(scene_names)

k = int(0.8 * len(scene_names))
train_scenes = scene_names[:k]
val_scenes   = scene_names[k:]

# ghi file
with open(os.path.join(SPLIT_DIR, "train_scenes.txt"), "w") as f:
    f.write("\n".join(train_scenes))
with open(os.path.join(SPLIT_DIR, "val_scenes.txt"), "w") as f:
    f.write("\n".join(val_scenes))

print(f"Done: {len(train_scenes)} train, {len(val_scenes)} val")
print(f"Saved at: {SPLIT_DIR}/train_scenes.txt, {SPLIT_DIR}/val_scenes.txt")
