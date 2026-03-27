# Dynamic Uncertainty-Aware BEV Occupancy Grid
### MAHE Hackathon 2026 - AI in Mobility - Problem Statement 3

## Project Overview
A two-stage pipeline that converts front camera images 
into metrically accurate Bird's Eye View occupancy grids
for Level 4 autonomous vehicles.

## Results
| Method | IoU Score |
|--------|-----------|
| Baseline | 0.0013 |
| Traditional IPM | 0.2747 |
| Neural Network | 0.4844 |
| Best Sample | 0.6008 |
| Improvement | 210x better |

## Model Architecture
- Stage 1: Real Inverse Perspective Mapping (IPM)
- Stage 2: Custom BEV Refinement CNN
- Encoder: Conv2D blocks (3->32->64->128)
- Decoder: ConvTranspose2D blocks (128->16->1)
- Parameters: 484,000+
- Loss: Binary Cross Entropy
- Optimizer: Adam

## Dataset
nuScenes Mini Dataset
- 15 scenes used
- 404 total samples
- 28,444 LiDAR points per frame
- Resolution: 1600x900

## Setup and Installation
pip install nuscenes-devkit torch opencv-python matplotlib scipy pyquaternion

## How to Run
1. Open BEV_Hackathon.ipynb in Google Colab
2. Mount Google Drive
3. Run all cells sequentially
4. View results automatically

## Live Demo
[Click here for Live Streamlit Demo](https://bevgrid-ai.streamlit.app/)

## Results Images
![Final Result](final_result.png)
![Comparison](final_comparison.png)
![Multi Sample](multi_sample_results.png)
![Training Loss](training_loss.png)

## Tech Stack
- Python 3.12
- PyTorch 2.x
- OpenCV
- nuScenes devkit
- Google Colab T4 GPU
- Streamlit (deployment)
