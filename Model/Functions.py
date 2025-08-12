import os
from typing import List
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

# Contrastive Loss Function (NT-Xent / InfoNCE)
def nt_xent_loss(video_embs, text_embs, temperature: float = 0.1):
    """
    video_embs: Tensor (B, D)   — L2-normalized video embeddings
    text_embs : Tensor (B, D)   — L2-normalized text embeddings
    returns   : scalar loss tensor
    """
    # Computing Similarity Matrix (B × B)
    sim_matrix = torch.mm(video_embs, text_embs.t())
    sim_matrix = sim_matrix / temperature

    # Labels are 0,1,...,B-1 so that sim_matrix[i,i] are positives
    batch_size = sim_matrix.size(0)
    labels = torch.arange(batch_size, device=sim_matrix.device)

    # Cross-entropy over rows (pulls diagonal up, pushes off-diagonals down)
    loss = nn.CrossEntropyLoss()(sim_matrix, labels)
    return loss


# Given a scene number, return the corresponding front camera image path from nuScenes.
def get_first_cam_front_image_path(nusc, scene_number: int) -> str:
    """
    nusc        : NuScenes instance
    scene_number: 1-based scene index as you present to the user
    returns     : absolute path to the first CAM_FRONT image
    """
    # Convert to zero-based index
    scene_index = scene_number - 1

    # Bound check
    if scene_index < 0 or scene_index >= len(nusc.scene):
        raise IndexError(f"Scene {scene_number} is out of range. Valid range is 1..{len(nusc.scene)}")

    # NuScenes lookups
    scene = nusc.scene[scene_index]
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    cam_front_token = sample['data']['CAM_FRONT']
    cam_front_data = nusc.get('sample_data', cam_front_token)

    # Build absolute path
    image_path = os.path.join(nusc.dataroot, cam_front_data['filename'])
    return image_path


# Collect ALL CAM_FRONT frame paths for a given scene.
def get_cam_front_sequence_paths(nusc, scene_number: int) -> List[str]:
    """
    nusc        : NuScenes instance
    scene_number: 1-based scene index
    returns     : list of absolute CAM_FRONT image paths (in order)
    """
    # Convert to zero-based index
    scene_index = scene_number - 1

    # Bound check
    if scene_index < 0 or scene_index >= len(nusc.scene):
        raise IndexError(f"Scene {scene_number} is out of range. Valid range is 1..{len(nusc.scene)}")

    # Walk the scene's linked list of samples
    scene = nusc.scene[scene_index]
    sample_token = scene['first_sample_token']

    cam_front_images = []
    while sample_token:
        sample = nusc.get('sample', sample_token)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        cam_path = os.path.join(nusc.dataroot, cam_data['filename'])
        cam_front_images.append(cam_path)
        sample_token = sample['next'] if sample['next'] else None

    return cam_front_images


# Display a single frame of an inputted Scene (non-interactive; good for scripts)
def display_scene(nusc, scene_number: int):
    """
    nusc        : NuScenes instance
    scene_number: 1-based scene index
    action      : opens the first CAM_FRONT frame and shows it via matplotlib
    """
    image_path = get_first_cam_front_image_path(nusc, scene_number)
    img = Image.open(image_path).convert("RGB")
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Scene {scene_number} - First CAM_FRONT Frame")
    plt.show()


# Preserve frames in a clip to bypass default_collate that PyTorch uses for DataLoader
def collate_fn(batch):
    """
    batch: list of (video_clip, instructions) where
           video_clip is Tensor (T_i, 3, 224, 224)
           instructions is list[str]
    returns: (videos_list, instrs_list) keeping variable-length T_i intact
    """
    videos = []
    instrs = []

    for video, instructions in batch:
        videos.append(video)          # video: Tensor (T_i, 3, 224, 224)
        instrs.append(instructions)   # instructions: list[str]

    return videos, instrs
