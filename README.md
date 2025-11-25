
<img width="1922" height="328" alt="image" src="https://github.com/user-attachments/assets/97d98ade-73e0-4b03-b93c-3c3e1431ab62" />


<img width="590" height="708" alt="image" src="https://github.com/user-attachments/assets/8bba3d95-1b3d-4a60-8701-2237fbcec501" />

<img width="889" height="500" alt="image" src="https://github.com/user-attachments/assets/45d6d416-ad0e-4dda-a52b-8d5f2221c05b" />

<img width="889" height="500" alt="image" src="https://github.com/user-attachments/assets/ef290136-3ecc-4431-958c-a91ff09adce0" />


# Neural-LinUCB Auto Exposure Control for oCam

> **2025 Introduction to Reinforcement Learning Project**

> **Topic:** Real-time Auto Exposure Control System using Reinforcement Learning (Neural-LinUCB)

## 1. Project Overview

This project addresses the instability of auto-exposure (AE) systems in robotics under drastic lighting changes. Traditional PID-based AE often fails to adapt quickly to abrupt transitions (e.g., lights turning on/off), leading to "blind" frames where visual information is lost.

To solve this, we implemented a **Neural-LinUCB (Deep Contextual Bandit)** algorithm. The agent learns to:

- **Analyze Image Features:** Extracts entropy, sharpness, and saturation from the camera feed.
- **Control Exposure Actively:** Selects the optimal exposure time to maximize image information and minimize control instability.
- **Overcome Hardware Limits:** Effectively handles physical shutter speed constraints in abrupt light transitioning environments.

## 2. Hardware & Software Requirements

### Hardware
- **Camera:** oCam-1MGN-U-T (Global Shutter USB Camera)
- **Compute:** Laptop or Embedded PC (Tested on Ubuntu 20.04)

### Software Dependencies
This project requires **Python 3.8+**.
Key libraries: PyTorch, OpenCV, NumPy, Pandas, Matplotlib

## 3. Installation

### Step 1. Clone the Repository
```bash
git clone [https://github.com/sdh123987-maker/Intoroduction-to-RL.git](https://github.com/sdh123987-maker/Intoroduction-to-RL.git)
cd Intoroduction-to-RL
```
### Step 2. Install Dependencies
```bash
pip install numpy opencv-python torch pandas matplotlib
```
## 4. How to Run

### Step 1. Connect Camera & Run Control Loop
Connect the oCam to your USB port and execute the main RL script.

Basic execution (Default Device ID: 2)

```bash
python RL_final.py
```
If your camera is at /dev/video0
```bash
python RL_final.py --device 0
```
### Arguments:

--device: Camera device index (default: 2)

--fps: Target FPS (default: 30)

--hz: Control loop frequency (default: 15)

### Step 2. Visualize Results
After stopping the script (Press ESC), a .csv log file will be generated. Use the plotting script to visualize the learning curves:

Note: Ensure the csv filename in the script matches your generated log file.

```bash
python plot.py
```
## 5. Key Features
Deep Representation Learning: Neural encoder maps high-dimensional image statistics to a latent state.

Contextual Bandit (LinUCB): Balances exploration and exploitation for exposure selection.

Safety Mechanism: Prevents penalty accumulation when hardware exposure limits are reached.

## 6. Trained Models

The trained agent consists of two key files. Both are required to load the model.

- **Encoder Weights (`.pth`):** Stores the parameters of the neural network that extracts features from images.
- **LinUCB Parameters (`.npz`):** Stores the learned contextual bandit matrices ($\theta, A^{-1}, b$) and action space.
- **Training Logs (`.csv`):** Contains step-by-step metrics (reward, exposure, image stats) for analysis.

> **Note:** To resume training or run the agent, ensure the timestamp in the filenames matches.

Author: Mingyu Kim Contact: sdh3163@naver.com
