# Dynamic Uncertainty-Aware BEV Occupancy Grid
MAHE Hackathon 2026 - AI in Mobility - Problem Statement 3

## Results
- Traditional IPM IoU: 0.2747
- Neural Network IoU: 0.4844
- Improvement: 76% better
- Best Sample IoU: 0.6008
- Samples Tested: 15

## Pipeline
1. Real Camera Calibration IPM
2. Neural Network Refinement CNN
3. LiDAR Ground Truth Validation
4. Risk Map Distance Weighted
5. Uncertainty Estimation

## Tech Stack
Python, PyTorch, OpenCV, nuScenes, Google Colab
