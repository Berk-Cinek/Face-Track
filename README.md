# C++ Face Tracking With OpenCV + ONNX runtime

![demo video](./screenshots/Adobe%20Express%20-%202c8871e5-6864-4025-bbff-cdcd10f2b078-render.gif)

A work-in-progress real-time head pose estimation pipeline using SCRFD face detection and SolvePnP to compute yaw, pitch, roll and camera distance from a webcam feed.

> Not yet configured for external builds — dependencies and model paths are local

## How it works:
* SCRFD detects face and returns 5 facial landmarks per frame
* Landmarks fed into SolvePnP with RANSAC for robust 3D pose solving
* Rodrigues transform converts rotation vector to Euler angles (pitch/yaw/roll)
* Rotation gating and linear smoothing stabilize output between frames

## Built with:
- **C++17**
- **ONNX Runtime** — CPU inference for SCRFD face detection model
- **OpenCV 4.x** — image preprocessing, letterboxing, NMS, pose axis visualization
- **spdlog** — structured logging to console and file


## Known limitations:
* Two-face instability
* pitch ambiguity at ±180°
* CPU only for now.