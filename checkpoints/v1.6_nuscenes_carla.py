# PARAMS
launcher = 'none'
load_from = 'checkpoints/AWML_centerpoint_v1.6_best_epoch_48.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
merge_objects = [
]
merge_type = 'extend_longer'
classes=[
    'car',
    'truck',
    'bus',
    'bicycle',
    'pedestrian',
]
train_batch_size = 6
val_batch_size = 4
work_dir = 'work_dirs/centerpoint_carla/'
lr=0.001
max_epoch=80

##
auto_scale_lr = dict(base_batch_size=64, enable=False)
backend_args = None
camera_panels = [
    'data/CAM_FRONT',
    'data/CAM_FRONT_RIGHT',
]
camera_types = {
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
}
class_colors = dict(
    barrier=(0, 0, 0,),
    bicycle=(255, 0, 30,),
    bus=(111,255,111),
    car=(30,144,255),
    construction_vehicle=(255,255,0),
    motorcycle=(100,0,30),
    pedestrian=(255,200,200),
    traffic_cone=(120,120,120),
    trailer=(0,255,255),
    truck=(140,0,255))
class_names = [
    'car',
    'truck',
    'bus',
    'bicycle',
    'pedestrian',
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.CenterPoint.models',
        'autoware_ml.detection3d.datasets.t4dataset',
        'autoware_ml.detection3d.evaluation.t4metric.t4metric',
        'autoware_ml.detection3d.datasets.transforms',
        'autoware_ml.hooks',
    ])
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=30,
        type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl', timeout=7200),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_class_range = dict(
    bicycle=121, bus=121, car=121, pedestrian=121, truck=121)
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        load_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=1,
        type='LoadPointsFromMultiSweeps',
        use_dim=[0,1,2,4,]),
    dict(
        point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
        type='PointsRangeFilter'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
filter_attributes = []

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            deterministic=True,
            max_num_points=32,
            max_voxels=(64000,64000,),
            point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
            voxel_size=[0.32,0.32,8.0])),
    pts_backbone=dict(
        conv_cfg=dict(bias=False, type='Conv2d'),
        in_channels=32,
        layer_nums=[3,5,5,],
        layer_strides=[1,2,2,],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[64,128,256,],
        type='SECOND'),
    pts_bbox_head=dict(
        bbox_coder=dict(
            code_size=9,
            max_num=500,
            out_size_factor=1,
            pc_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0],
            post_center_range=[-200.0,-200.0,-10.0,200.0,200.0,10.0,],
            score_threshold=0.1,
            type='CenterPointBBoxCoder',
            voxel_size=[0.32,0.32,8.0,]),
        common_heads=dict(
            dim=(3,2,),
            height=(1,2,),
            reg=(2,2,),
            rot=(2,2,),
            vel=(2,2,)),
        in_channels=384,
        loss_bbox=dict(
            loss_weight=0.25, reduction='mean', type='mmdet.L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, reduction='none', type='mmdet.GaussianFocalLoss'),
        norm_bbox=True,
        separate_head=dict(
            final_kernel=1, init_bias=-2.19, type='SeparateHead'),
        share_conv_channel=64,
        tasks=[
            dict(
                class_names=class_names,
                num_class=5),
        ],
        type='CenterHead'),
    pts_middle_encoder=dict(
        in_channels=32, output_shape=(760,760,), type='PointPillarsScatter'),
    pts_neck=dict(
        in_channels=[64,128,256,],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[128,128,128,],
        type='SECONDFPN',
        upsample_cfg=dict(bias=False, type='deconv'),
        upsample_strides=[1,2,4,],
        use_conv_for_no_stride=True),
    pts_voxel_encoder=dict(
        feat_channels=[32,32,],
        in_channels=4,
        legacy=False,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN1d'),
        point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
        type='BackwardPillarFeatureNet',
        voxel_size=[0.32,0.32,8.0,],
        with_cluster_center=True,
        with_distance=False,
        with_voxel_center=True),
    test_cfg=dict(
        pts=dict(
            grid_size=[760,760,1,],
            min_radius=[1.0,],
            nms_type='circle',
            out_size_factor=1,
            pc_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
            post_center_limit_range=[-200.0,-200.0,-10.0,200.0,200.0,10.0,],
            post_max_size=100,
            voxel_size=[0.32,0.32,8.0,])),
    train_cfg=dict(
        pts=dict(
            code_weights=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.2,0.2,],
            dense_reg=1,
            gaussian_overlap=0.1,
            grid_size=[760,760,1,],
            max_objs=500,
            min_radius=2,
            out_size_factor=1,
            point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
            voxel_size=[0.32,0.32,8.0,])),
    type='CenterPoint')
name_mapping = dict(
    {
    'ambulance': 'car',
    'animal': 'animal',
    'bicycle': 'bicycle',
    'bus': 'bus',
    'car': 'car',
    'construction_worker': 'pedestrian',
    'fire_truck': 'truck',
    'forklift': 'car',
    'kart': 'car',
    'motorcycle': 'bicycle',
    'movable_object.barrier': 'barrier',
    'movable_object.debris': 'debris',
    'movable_object.pushable_pullable': 'pushable_pullable',
    'movable_object.traffic_cone': 'traffic_cone',
    'movable_object.trafficcone': 'traffic_cone',
    'pedestrian': 'pedestrian',
    'pedestrian.adult': 'pedestrian',
    'pedestrian.child': 'pedestrian',
    'pedestrian.construction_worker': 'pedestrian',
    'pedestrian.personal_mobility': 'pedestrian',
    'pedestrian.police_officer': 'pedestrian',
    'pedestrian.stroller': 'pedestrian',
    'pedestrian.wheelchair': 'pedestrian',
    'personal_mobility': 'pedestrian',
    'police_car': 'car',
    'police_officer': 'pedestrian',
    'semi_trailer': 'trailer',
    'static_object.bicycle rack': 'bicycle rack',
    'static_object.bicycle_rack': 'bicycle_rack',
    'static_object.bollard': 'bollard',
    'stroller': 'pedestrian',
    'tractor_unit': 'truck',
    'trailer': 'truck',
    'truck': 'truck',
    'vehicle.ambulance': 'car',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus': 'bus',
    'vehicle.bus (bendy & rigid)': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'truck',
    'vehicle.emergency (ambulance & police)': 'car',
    'vehicle.fire': 'truck',
    'vehicle.motorcycle': 'bicycle',
    'vehicle.police': 'car',
    'vehicle.trailer': 'truck',
    'vehicle.truck': 'truck',
    'wheelchair': 'pedestrian'
}
    )
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=lr, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
out_size_factor = 1
param_scheduler = [
    dict(
        T_max=10,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        eta_min=0.0029999999999999996,
        type='CosineAnnealingLR'),
    dict(
        T_max=35,
        begin=15,
        by_epoch=True,
        convert_to_iter_based=True,
        end=max_epoch,
        eta_min=3e-08,
        type='CosineAnnealingLR'),
    dict(
        T_max=10,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        eta_min=0.8947368421052632,
        type='CosineAnnealingMomentum'),
    dict(
        T_max=35,
        begin=15,
        by_epoch=True,
        convert_to_iter_based=True,
        end=max_epoch,
        eta_min=1,
        type='CosineAnnealingMomentum'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=0)
sync_bn = 'torch'
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        class_names=class_names,
        data_prefix=dict(pts="samples/LIDAR_TOP", img="", sweeps="samples/LIDAR_TOP"),
        data_root='data/nuscenes/',
        metainfo=dict(classes=classes),
        modality=dict(
            use_camera=False,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                load_dim=5,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=1,
                type='LoadPointsFromMultiSweeps',
                use_dim=[0,1,2,4,]),
            dict(
                point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
                type='PointsRangeFilter'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='T4Dataset'),
    num_workers = 12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes',
    metric='bbox',
    type='NuScenesMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        load_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=1,
        type='LoadPointsFromMultiSweeps',
        use_dim=[0,1,2,4,]),
    dict(
        point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
        type='PointsRangeFilter'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
train_cfg = dict(
    by_epoch=True,
    dynamic_intervals=[(45,1,)],
    max_epochs=max_epoch,
    val_interval=1)
train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(
        ann_file='nuscenes_infos_train.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        class_names=class_names,
        data_prefix=dict(pts="samples/LIDAR_TOP", img="", sweeps="samples/LIDAR_TOP"),
        data_root='data/nuscenes/',
        metainfo=dict(classes=classes),
        modality=dict(
            use_camera=False,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                load_dim=5,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=1,
                type='LoadPointsFromMultiSweeps',
                use_dim=[0,1,2,4,]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5,
                sync_2d=False,
                type='RandomFlip3D'),
            dict(
                rot_range=[
                    -1.571,
                    1.571,
                ],
                scale_ratio_range=[
                    0.8,
                    1.2,
                ],
                translation_std=[
                    1.0,
                    1.0,
                    0.2,
                ],
                type='GlobalRotScaleTrans'),
            dict(
                point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
                type='ObjectRangeFilter'),
            dict(
                classes=classes,
                type='ObjectNameFilter'),
            dict(min_num_points=5, type='ObjectMinPointsFilter'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='T4Dataset'),
    num_workers = 1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        load_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=1,
        type='LoadPointsFromMultiSweeps',
        use_dim=[0,1,2,4,]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        sync_2d=False,
        type='RandomFlip3D'),
    dict(
        rot_range=[
            -1.571,
            1.571,
        ],
        scale_ratio_range=[
            0.8,
            1.2,
        ],
        translation_std=[
            1.0,
            1.0,
            0.2,
        ],
        type='GlobalRotScaleTrans'),
    dict(
        point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
        type='ObjectRangeFilter'),
    dict(
        classes=classes,
        type='ObjectNameFilter'),
    dict(min_num_points=5, type='ObjectMinPointsFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=val_batch_size,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        class_names=class_names,
        data_prefix=dict(pts="samples/LIDAR_TOP", img="", sweeps="samples/LIDAR_TOP"),
        data_root='data/nuscenes/',
        metainfo=dict(classes=classes),
        modality=dict(
            use_camera=False,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                load_dim=5,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=1,
                type='LoadPointsFromMultiSweeps',
                use_dim=[0,1,2,4,]),
            dict(
                point_cloud_range=[-121.6,-121.6,-3.0,121.6,121.6,5.0,],
                type='PointsRangeFilter'),
            dict(
                keys=[
                    'points',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='T4Dataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
