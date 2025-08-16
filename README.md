# Pose Detection with Masked Human Region (MediaPipe + OpenCV)

This project is an extension of my previous [pose-detection](https://github.com/dipan313/PoseEstimation) repo.  
Along with detecting human pose landmarks using **MediaPipe**, this version also applies **Selfie Segmentation** to generate a mask.  
The human region is blacked out for **privacy/security**, while landmarks are still drawn.

---

## Features
- Real-time human pose detection
- Background segmentation with **masked human**
- Privacy-preserving view (human area = blacked out)
- FPS counter

---

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
---
## Run
```bash
python masked_pose_detection.py
```

