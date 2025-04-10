# AetherManipulator

## Description
An interactive Python application using MediaPipe to track hand gestures in real-time, allowing users to manipulate 3D models on screen as if handling holographic projections, you know, for science!

## Features
- Real-time hand landmark detection via webcam using MediaPipe Hands.
- Gesture mapping: Translate models by moving an open hand, rotate models using a closed fist gesture (like holding an object), and scale using a two-handed pinch/spread gesture.
- Interactive rendering of basic 3D models (e.g., OBJ format) within an OpenCV window using PyOpenGL.
- Intuitive control translating 2D hand movements from the camera feed into 3D model transformations (translation, rotation, scaling).
- Basic UI elements (using OpenCV drawing functions) to display current mode (translate/rotate/scale) or loaded model name.

## Learning Benefits
Gain hands-on experience with real-time computer vision (MediaPipe), gesture recognition logic, integrating ML outputs with graphical rendering (OpenGL), understanding basic 3D coordinate transformations, and building interactive ML applications. It's basically building your own 'proof of concept' JARVIS interface for 3D visualization.

## Technologies Used
- opencv-python
- mediapipe
- numpy
- PyOpenGL
- PyOpenGL_accelerate
- pywavefront

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/aethermanipulator.git
cd aethermanipulator

# Install dependencies
pip install -r requirements.txt
```

## Usage
[Instructions on how to use the project]

## Project Structure
[Brief explanation of the project structure]

## License
MIT

## Created with AI
This project was automatically generated using an AI-powered project generator.
