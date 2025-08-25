import os
from typing import List
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

# Contrastive Loss Function
def nt_xent_loss(v: torch.Tensor, t: torch.Tensor, tau: float, symmetric: bool):
    """
    Strict NT-Xent for one positive per anchor.
    v, t: (B, D) L2-normalized embeddings (same B).
    tau: temperature.
    symmetric=True averages video→text and text→video; set False for one direction.
    """
    if v.shape != t.shape:
        raise ValueError(f"Shape mismatch: v {v.shape} vs t {t.shape}")
    if v.size(0) < 2:
        raise ValueError("Batch size must be >= 2 for NT-Xent.")

    # Cosine-sim logits scaled by temperature
    logits = (v @ t.T) / tau                      # (B, B)

    # Ground-truth: diagonal is the ONLY positive (strict NT-Xent)
    labels = torch.arange(logits.size(0), device=logits.device)

    if not symmetric:
        # Non Symmetric NT-Xent (video->text and text->video)
        return F.cross_entropy(logits, labels)
    else:
      # Symmetric NT-Xent (video->text and text->video)
      return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

# Given a scene number, return the corresponding front camera image path from nuScenes.
def get_first_cam_front_image_path(scene_number: int):
    scene_index = scene_number - 1

    scene = nusc.scene[scene_index]

    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    cam_front_token = sample['data']['CAM_FRONT']
    cam_front_data = nusc.get('sample_data', cam_front_token)

    image_path = os.path.join(nusc.dataroot, cam_front_data['filename'])
    return image_path


# Display a single frame of inputted Scene
def display_scene():
  try:
    scene_number = int(input("Enter a scene number: "))
    image_path = get_first_cam_front_image_path(scene_number)
  except (IndexError, ValueError):
    print("Invalid Scene Selection! Please enter a number between 1 and 10")
  else:
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Scene {scene_number} - First CAM_FRONT Frame")
    plt.show()


# Display all frames of an inputted scene
def view_cam_front_sequence():
    try:
        scene_number = int(input("Enter a scene number: "))
        if scene_number < 1 or scene_number > 10:
            raise IndexError("Scene number out of bounds.")

        scene = nusc.scene[scene_number - 1]
        sample_token = scene['first_sample_token']

        cam_front_images = []

        while sample_token:
            sample = nusc.get('sample', sample_token)
            cam_token = sample['data']['CAM_FRONT']
            cam_data = nusc.get('sample_data', cam_token)
            cam_path = os.path.join(nusc.dataroot, cam_data['filename'])
            cam_front_images.append(cam_path)
            sample_token = sample['next'] if sample['next'] else None

        print(f"Loaded {len(cam_front_images)} CAM_FRONT frames for Scene {scene_number}")

        # Interactive display
        index = widgets.IntSlider(min=0, max=len(cam_front_images)-1, step=1, description="Frame:")

        def show_frame(i):
            clear_output(wait=True)
            img = Image.open(cam_front_images[i])
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Scene {scene_number} - Frame {i+1}/{len(cam_front_images)}")
            plt.show()
            display(index)

        index.observe(lambda change: show_frame(change.new), names='value')
        show_frame(0)

    except (ValueError, IndexError):
        print("Invalid Scene Selection. Please enter a number between 1 and 10")

# Preserve frames in a clip to bypass default_collate that pytorch uses for DataLoader
def collate_fn(batch):
    videos = []
    instrs = []
    for video, instructions in batch:
        videos.append(video)         # video: Tensor (T_i,3,224,224)
        instrs.append(instructions)  # instructions: list of str
    return videos, instrs

# Choose the single instruction used as the positive for NT-Xent
def pick_instruction(texts: str):
    if not texts:
        return "maintain normal driving motion"
    return texts[-1]  # last instruction aligns with your 'final frames' rational

def l2n(x: int):
    # L2 normalize with an epsilon guard
    return F.normalize(x, p=2, dim=-1, eps=eps)