### build
DOCKER_BUILDKIT=1 docker build -t autoware-ml .

### run
docker run -it --rm --gpus '"device=0"' --shm-size=64g --name awml -p 6006:6006 -v $PWD/:/workspace -v $PWD/data:/workspace/data autoware-ml

Or run in devcontainer

### convert
```
python tools/detection3d/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

### train
```
python tools/detection3d/train.py checkpoints/v1.4_nuscenes.py
```

### test
```
python tools/detection3d/test.py checkpoints/v1.4_nuscenes.py checkpoints/centerpoint_v1.4_best_epoch.pth
```

### visualize
#### test
```
python tools/detection3d/test.py work_dirs/ntt_centerpoint_v1.4/v1.4_nuscenes.py work_dirs/ntt_centerpoint_v1.4/epoch_50.pth --show --task lidar_det
```

#### bev
```
python tools/detection3d/visualize_bev.py work_dirs/ntt_centerpoint_v1.4/v1.4_nuscenes.py --checkpoint work_dirs/ntt_centerpoint_v1.4/epoch_50.pth 
```
optional `--show-gt` tag to show ground truth instead of predictions