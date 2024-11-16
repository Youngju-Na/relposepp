import torch

from dataset.co3d_v2 import Co3dDataset

from omegaconf import OmegaConf
train_dataset_cfg = {
    'view_sampler': 
        {'name': 'bounded', 
         'num_target_views': 1, 
         'num_context_views': 3, 
         'min_distance_between_context_views': 2, 
         'max_distance_between_context_views': 6, 
         'min_distance_to_context_views': 0, 
         'warm_up_steps': 0, 
         'initial_min_distance_between_context_views': 2, 
         'initial_max_distance_between_context_views': 6
         }, 
    'name': 'dtu', 
    'roots': ['/home/youngju/3d-recon/datasets/DTU'], 
    'pair_filepath': '/home/youngju/3d-recon/relpose-plus-plus/relpose/src/dataset/dtu/dtu_pairs.txt', 
    'split_filepath': ['/home/youngju/3d-recon/relpose-plus-plus/relpose/src/dataset/dtu/lists'], 
    'make_baseline_1': False, 
    'augment': True, 
    'n_views': 10, 
    'num_context_views': 3, 
    'num_all_imgs': 49, 
    'test_context_views': [23, 24, 33], 
    'test_target_views': [35, 25], 
    'single_view': False, 
    'view_selection_type': 
    'random', 
    'image_shape': [224, 224], 
    'original_image_shape': [128, 160], 
    'background_color': [0.0, 0.0, 0.0], 
    'cameras_are_circular': False, 
    'baseline_epsilon': 0.001, 
    'max_fov': 100.0, 
    'skip_bad_shape': True, 
    'near': 1.0, 
    'far': 100.0, 
    'baseline_scale_bounds': False, 
    'shuffle_val': True, 
    'test_len': -1, 
    'test_chunk_interval': 1, 
    'overfit_to_scene': None
}
train_dataset_cfg = OmegaConf.create(train_dataset_cfg)
test_datsset_cfg = {
    'view_sampler': {
        'name': 'evaluation', 
        'index_path': '/home/youngju/3d-recon/relpose-plus-plus/relpose/assets/evaluation_index_re10k_video.json', 
        'num_context_views': 3
    }, 
    'name': 'dtu', 
    'roots': ['/home/youngju/3d-recon/datasets/DTU'], 
    'pair_filepath': '/home/youngju/3d-recon/relpose-plus-plus/relpose/src/dataset/dtu/dtu_pairs.txt', 
    'split_filepath': ['/home/youngju/3d-recon/relpose-plus-plus/relpose/src/dataset/dtu/lists'], 
    'make_baseline_1': False, 
    'augment': True, 
    'n_views': 10, 
    'num_context_views': 3, 
    'num_all_imgs': 49, 
    'test_context_views': [23, 24, 33], 
    'test_target_views': [22, 15, 34], 
    'single_view': False, 
    'view_selection_type': 'random', 
    'image_shape': [224, 224], 
    'original_image_shape': [128, 160], 
    'background_color': [0.0, 0.0, 0.0], 
    'cameras_are_circular': False, 
    'baseline_epsilon': 0.001, 
    'max_fov': 100.0, 
    'skip_bad_shape': True, 
    'near': 1.0, 
    'far': 100.0, 
    'baseline_scale_bounds': False, 
    'shuffle_val': True, 
    'test_len': -1, 
    'test_chunk_interval': 1, 
    'overfit_to_scene': None
}
test_datsset_cfg = OmegaConf.create(test_datsset_cfg)

def get_dataloader(
    batch_size=64,
    dataset="co3d",
    category=("table",),
    split="train",
    shuffle=True,
    num_workers=8,
    debug=False,
    num_images=2,
    rank=None,
    world_size=None,
    img_size=224,
    normalize_cameras=False,
    random_num_images=False,
    first_camera_transform=False,
    first_camera_rotation_only=False,
    mask_images=False,
):
    if debug:
        num_workers = 0
    if dataset == "co3d":
        dataset = Co3dDataset(
            category=category,
            split=split,
            num_images=num_images,
            debug=debug,
            img_size=img_size,
            normalize_cameras=normalize_cameras,
            random_num_images=random_num_images,
            first_camera_transform=first_camera_transform,
            first_camera_rotation_only=first_camera_rotation_only,
            mask_images=mask_images,
        )
    elif dataset == "dtu":
        from src.dataset.dataset_dtu import DatasetDTU
        from src.dataset.view_sampler import get_view_sampler
        from src.misc.step_tracker import StepTracker
        step_tracker = StepTracker()   
        stage="train"
        view_sampler = get_view_sampler(
            train_dataset_cfg.view_sampler,
            "train",
            train_dataset_cfg.overfit_to_scene is not None,
            train_dataset_cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DatasetDTU(
            cfg=train_dataset_cfg,
            stage=stage,
            view_sampler=view_sampler,
            normalize_cameras=normalize_cameras,
            num_images=num_images,
            img_size=img_size,
            first_camera_transform=first_camera_transform,
            first_camera_rotation_only=first_camera_rotation_only,
            mask_images=mask_images,
        )
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    if rank is None:
        sampler = None
    else:
        if world_size is None:
            assert False
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        print(f"Sampler {rank} {world_size}")
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )
