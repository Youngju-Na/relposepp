import numpy as np

from dataset.co3d_v2 import Co3dDataset
from src.dataset.dataset_dtu import DatasetDTU
from omegaconf import OmegaConf
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
    'test_context_views': [34, 14, 32], 
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

def compute_angular_error(rotation1, rotation2):
    R_rel = rotation1.T @ rotation2
    tr = (np.trace(R_rel) - 1) / 2
    theta = np.arccos(tr.clip(-1, 1))
    return theta * 180 / np.pi


def compute_angular_error_batch(rotation1, rotation2):
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi


def get_dataset(
    category,
    num_images,
    split="test",
    eval_time=True,
    normalize_cameras=False,
    first_camera_transform=False,
    dataset='co3d',
):
    if dataset == 'co3d':
        dataset = Co3dDataset(
            split=split,
            category=[category],
            num_images=num_images,
            eval_time=eval_time,
            normalize_cameras=normalize_cameras,
            first_camera_transform=first_camera_transform,
        )
    elif dataset == 'dtu':
        from src.dataset.dataset_dtu import DatasetDTU
        from src.dataset.view_sampler import get_view_sampler
        from src.misc.step_tracker import StepTracker
        step_tracker = StepTracker()   
        view_sampler = get_view_sampler(
            test_datsset_cfg.view_sampler,
            "test",
            test_datsset_cfg.overfit_to_scene is not None,
            test_datsset_cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DatasetDTU(
            cfg=test_datsset_cfg,
            stage='test',
            view_sampler=view_sampler,
            normalize_cameras=normalize_cameras,
            eval_time=eval_time,
            first_camera_transform=first_camera_transform,
            num_images=num_images,
        )

    return dataset
