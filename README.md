# Contrastive Learning for Language-Guided Routing in Autonomous Vehicles

This project explores how autonomous vehicles can better understand and respond to human instructions by combining natural language, visual scenes, and motion data. We use contrastive learning to align these three forms of input—so that when someone gives a command like “turn left after the stop sign,” the vehicle can interpret that instruction in context and relate it to the scene and appropriate movement.

## Goals

Our goal is to build a model that can understand how language, visual context, and motion relate in real-world driving scenarios. We aim to enable autonomous systems to interpret flexible human instructions and align them with their surrounding environment and behavior. This includes developing a tri-modal contrastive learning model and integrating automated data curation tools like ADVLAT to scale our training process and improve model generalization.

## Data Sources

- **doScenes** doScenes — natural-language instructions and annotations.
- **nuScenes** 
  
## ADVLAT Engine Integration

In addition to the contrastive model, we're replicating the **ADVLAT Engine**, an automated data curation system that uses GPS, video, and NLP to generate structured (scene, instruction, trajectory) triads without manual labeling. Integrating ADVLAT helps to generate customized scene-instruction-trajectory triads at scale allowing us to synthesize additional training and evaluation data on demand, improving robustness and adaptability of our contrastive model beyond the original doScenes coverage.
