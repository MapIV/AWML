import pickle

def count_classes(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    categories = data["metainfo"].get("categories") or data["metainfo"].get("classes")
    
    # Chuẩn hoá thành id2name
    if isinstance(categories, dict):
        # {class_name: index}
        id2name = {v: k for k, v in categories.items() if v >= 0}
    elif isinstance(categories, (list, tuple)):
        id2name = {i: k for i, k in enumerate(categories)}
    else:
        raise ValueError("Unknown categories format")

    counts = {idx: {"name": name, "count": 0} for idx, name in id2name.items()}
    for item in data["data_list"]:
        for inst in item.get("instances", []):
            lbl = inst.get("bbox_label_3d", -1)
            if lbl in id2name:
                counts[lbl]["count"] += 1
    return counts

train_counts = count_classes("/workspace/data/nuscenes/nuscenes_infos_train.pkl")
val_counts = count_classes("/workspace/data/nuscenes/nuscenes_infos_val.pkl")

print("Train:")
for idx, info in train_counts.items():
    print(f"  id={idx}, name={info['name']}, count={info['count']}")

print("Val:")
for idx, info in val_counts.items():
    print(f"  id={idx}, name={info['name']}, count={info['count']}")
