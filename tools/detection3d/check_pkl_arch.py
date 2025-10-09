import pickle
import json
import numpy as np

def numpy_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
path = "/workspace/data/nuscenes/nuscenes_infos_train.pkl"   # đổi thành file của bạn
json_out = "/workspace/data/nuscenes/nuscenes_infos_train_dump.json"




with open(path, "rb") as f:
    data = pickle.load(f)
    
# dump ra JSON với indent để dễ đọc
with open(json_out, "w") as f:
    json.dump(data, f, indent=2, default=numpy_encoder)
    
print(f"✅ Dumped to {json_out}")

print("Type of data:", type(data))
if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
    for k, v in data.items():
        if isinstance(v, (list, tuple)):
            print(f"  {k}: list of length {len(v)}")
            if v:  # lấy phần tử đầu tiên để xem cấu trúc
                print(f"    First element type: {type(v[0])}")
                if isinstance(v[0], dict):
                    print(f"    Keys of first element: {list(v[0].keys())}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")
        else:
            print(f"  {k}: {type(v)}")
else:
    print("Not a dict, type:", type(data))
