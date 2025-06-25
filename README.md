# Contrastive Learning for Language-Guided Routing in Autonomous Vehicles

This project explores how autonomous vehicles can better understand and respond to human instructions by combining natural language, visual scenes, and motion data. We use contrastive learning to align these three forms of input—so that when someone gives a command like “turn left after the stop sign,” the vehicle can interpret that instruction in context and relate it to the scene and appropriate movement.

Current autonomous driving systems often rely on hard-coded commands or simple object detection. Our approach brings a more human-centered way of communicating with vehicles, making it possible for a car to understand flexible language and respond more safely in dynamic, real-world scenarios. By training the model to recognize when all three inputs refer to the same real-world situation, it can learn to generalize and respond to new instructions or unfamiliar environments.

## Data Sources

- **doScenes** – 
- **nuScenes** –
  
## ADVLAT Engine Integration

In addition to the contrastive model, we're replicating the **ADVLAT Engine**, an automated data curation system that uses GPS, video, and NLP to generate structured (scene, instruction, trajectory) triads without manual labeling. Integrating ADVLAT helps to generate customized scene-instruction-trajectory triads at scale allowing us to synthesize additional training and evaluation data on demand, improving robustness and adaptability of our contrastive model beyond the original doScenes coverage.
