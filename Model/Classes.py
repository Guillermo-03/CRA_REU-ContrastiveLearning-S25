import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import pandas as pd


# Defining Dataset
class SceneLevelDataset(Dataset):
    def __init__(self, nusc, annotations_df):
      self.nusc = nusc

      # Mapping Instructions to Scenes
      # For scenes with multiple annotations, it looks like: [instruction1, instruction2, …]
      self.scene_to_instructions = {}

      for _, row in annotations_df.iterrows():
          sn = int(row['Scene Number'])
          instr = row['Instruction']

          if sn not in self.scene_to_instructions:
              self.scene_to_instructions[sn] = []

          self.scene_to_instructions[sn].append(instr)

      # Replacing any NaN-only lists with a default instruction
      default_instr = "maintain normal driving motion"
      for sn, instr_list in self.scene_to_instructions.items():
          # Filter out non-string or NaN entries
          cleaned = [inst for inst in instr_list if isinstance(inst, str) and not pd.isna(inst)]
        
          # If nothing left after cleaning, use default
          if not cleaned:
              cleaned = [default_instr]
          self.scene_to_instructions[sn] = cleaned

      # Creating a list of scenes that have an annotation
      self.scene_numbers = list(self.scene_to_instructions.keys())

      # Creating lookup tables for quick data access
      self.scene_num_to_token = {}
      self.first_tokens = {}

      for scene in self.nusc.scene:
          name = scene['name']
          parts = name.split('-')

          try:
              num = int(parts[-1])
              token = scene['token']
              self.scene_num_to_token[num] = token
                # storing first sample token
              rec = self.nusc.get('scene', token)
              self.first_tokens[token] = rec['first_sample_token']
          except ValueError:
              continue

      # Image transform
      self.transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
      ])

    def __len__(self):
      return len(self.scene_numbers)

    def __getitem__(self, idx):
      scene_num = self.scene_numbers[idx]
      token = self.scene_num_to_token.get(scene_num)

      if token is None:
          raise IndexError(f"Scene {scene_num} not in nuScenes-mini")

        # Walk all front‐cam sample tokens in this scene
      sample_tokens = []
      curr = self.first_tokens[token]

      while curr:
        sample_tokens.append(curr)
        rec = self.nusc.get('sample', curr)
        curr = rec['next'] if rec['next'] else None

        # Load all frames
      frames = []
      for st in sample_tokens:
        samp = self.nusc.get('sample', st)
        cam_tok = samp['data']['CAM_FRONT']
        data = self.nusc.get('sample_data', cam_tok)
        path = os.path.join(self.nusc.dataroot, data['filename'])
        img = Image.open(path).convert('RGB')
        frames.append(self.transform(img))

        # Stack frames into a clip tensor (T, 3, 224, 224)
      video = torch.stack(frames)
      instructions = self.scene_to_instructions[scene_num]  # list of strings
      return video, instructions


# Defining Model
# Maps scene video and natural language to the same embedding space
class BiModalSequenceModel(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # Load ImageNet-pretrained ResNet-18.
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Removing final classification layer and keeping everything up to global avg-pool.
        modules = list(resnet.children())[:-1]   # all layers except the last FC
        self.backbone = nn.Sequential(*modules)

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Projection head for video features
        in_features = resnet.fc.in_features       # the dimension of the pooled feature
        self.image_proj = nn.Linear(in_features, embed_dim)

        # Text encoder: DistilBERT and projection
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = AutoModel.from_pretrained('distilbert-base-uncased')

        # Projection head for text features to shared space
        text_hidden = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_hidden, embed_dim)

        # L2 normalization
        self.normalize = nn.functional.normalize

    def encode_video(self, video):
        """
        video:  Tensor (B, T, 3, H, W)   — batch of scene clips (all frames)
        return: Tensor (B, D)            — L2-normalized scene embeddings
        B = Batch Size
        T = Total # of frames in scene
        D  = Embedding Dimensions
        """
        B, T, C, H, W = video.shape

        # Collapse (B, T) so we can feed frames through ResNet in a single pass
        video = video.view(B * T, C, H, W)
        feats = self.backbone(video)               # (B*T, in_features, 1, 1)
        feats = feats.view(B, T, -1)               # (B, T, in_features)

        # Average pool over time dimension
        pooled = feats.mean(dim=1)                 # (B, in_features)

        # Projection
        z = self.image_proj(pooled)                # (B, embed_dim)
        z = self.normalize(z, dim=-1)
        return z

    def encode_text(self, texts):
        """
        texts: list of strings of length B (or B separate calls)
        returns: Tensor of shape (len(texts), embed_dim), L2-normalized
        """
        # Tokenize and batch
        tokens = self.tokenizer(texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True)
        # Move tokens to same device as the model
        device = next(self.parameters()).device

        for key, tensor in tokens.items():
          tokens[key] = tensor.to(device)

        out = self.text_encoder(**tokens)
        # Use the [CLS] token embedding (first token) as sentence embedding
        pooled = out.last_hidden_state[:, 0, :]     # (B, hidden_size)

        # Projection
        z = self.text_proj(pooled)                  # (B, embed_dim)
        z = self.normalize(z, dim=-1)
        return z

    def forward(self, video, texts):
        """
        video: Tensor (B, T, 3, H, W)
        texts: list of B strings
        returns: (video_embeddings, text_embeddings)
        both of shape (B, embed_dim)
        """
        v = self.encode_video(video)
        t = self.encode_text(texts)
        return v, t