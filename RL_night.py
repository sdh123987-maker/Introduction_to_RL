#!/usr/bin/env python3
"""
Neural-LinUCB based Auto Exposure Control System for oCam-1MGN-U-T
** Night Mode Tuned (Conservative / 30FPS Lock) **
"""

import cv2
import numpy as np
import time
import csv
import os
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# 1. Online Statistics (Welford's Algorithm)
# ============================================================================

class WelfordStats:
    """Tracks running mean and standard deviation."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    def get_mean(self):
        return self.mean
    
    def get_std(self):
        if self.n < 2:
            return 1.0
        return np.sqrt(self.M2 / self.n)


# ============================================================================
# 2. Feature Extraction & Normalization
# ============================================================================

class FeatureExtractor:
    def __init__(self):
        self.stats = {}
        self.prev_mu = None
        self.warmup_threshold = 30
        
    def extract_features(self, frame, exp):
        # Basic statistics
        mu = float(np.mean(frame))
        sigma = float(np.std(frame))
        
        # 1. Histogram Entropy (Information content)
        hist = cv2.calcHist([frame], [0], None, [64], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        H = -np.sum(hist * np.log(hist + 1e-8))
        
        # 2. Image Sharpness (Laplacian Variance)
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        sharpness = float(laplacian.var())
        
        # 3. Saturation Ratio
        sat_hi = float(np.sum(frame >= 253) / frame.size)
        sat_lo = float(np.sum(frame <= 2) / frame.size)
        
        # 4. Flicker detection (Temporal brightness change)
        delta_mu = 0.0 if self.prev_mu is None else abs(mu - self.prev_mu)
        self.prev_mu = mu
        
        raw_features = {
            'mu': mu,
            'sigma': sigma,
            'H': H,
            'sharpness': sharpness,
            'sat_hi': sat_hi,
            'sat_lo': sat_lo,
            'delta_mu': delta_mu,
            'exp': float(exp)
        }
        
        # Online Normalization
        for key, val in raw_features.items():
            if key not in self.stats:
                self.stats[key] = WelfordStats()
            self.stats[key].update(val)
        
        normalized_features = {}
        for key, val in raw_features.items():
            if self.stats[key].n < self.warmup_threshold:
                normalized_features[key] = val
            else:
                mean = self.stats[key].get_mean()
                std = self.stats[key].get_std()
                normalized_features[key] = (val - mean) / (std + 1e-8)
        
        return normalized_features, raw_features


# ============================================================================
# 3. Neural Feature Encoder
# ============================================================================

class FeatureEncoder(nn.Module):
    """Maps high-dimensional image features to latent representation."""
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        # Auxiliary head for representation learning (predicting reward)
        self.reward_head = nn.Linear(output_dim, 1)
    
    def forward(self, x, return_reward=False):
        features = self.encoder(x)
        if return_reward:
            reward_pred = self.reward_head(features)
            return features, reward_pred
        return features


# ============================================================================
# 4. Neural-LinUCB Agent
# ============================================================================

@dataclass
class NeuralLinUCBConfig:
    feature_dim: int = 32
    lambda_reg: float = 0.01
    alpha: float = 1.5  # Exploration parameter
    unc_cap: float = 10.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    update_every: int = 10
    replay_size: int = 2000


class NeuralLinUCB:
    """
    Combines Deep Representation Learning with Linear UCB.
    """
    def __init__(self, input_dim=8, actions=None, config=None):
        self.config = config or NeuralLinUCBConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = FeatureEncoder(
            input_dim=input_dim, 
            output_dim=self.config.feature_dim
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.encoder.parameters(), 
            lr=self.config.learning_rate
        )
        
        self.actions = actions or [0]
        
        # LinUCB Parameters: Inverse Covariance (A_inv), Bias (b), Weights (theta)
        self.A_inv = {}
        self.b = {}
        self.theta = {}
        
        for action in self.actions:
            self.init_action_params(action)
        
        self.replay = deque(maxlen=self.config.replay_size)
        self.step_count = 0
    
    def init_action_params(self, action):
        d = self.config.feature_dim
        self.A_inv[action] = np.eye(d) / self.config.lambda_reg
        self.b[action] = np.zeros(d)
        self.theta[action] = np.zeros(d)
    
    def encode_features(self, features):
        feature_keys = ['mu', 'sigma', 'H', 'sharpness', 'sat_hi', 
                       'sat_lo', 'delta_mu', 'exp']
        x = np.array([features[k] for k in feature_keys], dtype=np.float32)
        
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
            phi = self.encoder(x_tensor).squeeze(0).cpu().numpy()
        
        return phi
    
    def select_action(self, features):
        """Select action using UCB (Upper Confidence Bound)."""
        phi = self.encode_features(features)
        
        best_action = None
        best_score = -float('inf')
        action_info = {}
        
        for action in self.actions:
            # Linear Bandit Prediction: Mean + Alpha * Uncertainty
            mean = np.dot(phi, self.theta[action])
            unc = np.sqrt(np.dot(phi, np.dot(self.A_inv[action], phi)))
            unc = np.clip(unc, 0, self.config.unc_cap)
            score = mean + self.config.alpha * unc
            
            action_info[action] = {
                'mean': mean,
                'unc': unc,
                'score': score
            }
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action, phi, action_info[best_action], features
    
    def update(self, phi, action, reward, raw_features):
        """Update LinUCB parameters using Sherman-Morrison formula."""
        A_inv = self.A_inv[action]
        b = self.b[action]
        
        A_inv_phi = np.dot(A_inv, phi)
        denominator = 1.0 + np.dot(phi, A_inv_phi)
        
        if denominator < 1e-6:
            denominator = 1e-6
        
        # Recursive Least Squares update
        self.A_inv[action] = A_inv - np.outer(A_inv_phi, A_inv_phi) / denominator
        self.b[action] = b + reward * phi
        self.theta[action] = np.dot(self.A_inv[action], self.b[action])
        
        self.replay.append((raw_features, action, reward))
    
    def train_encoder(self):
        """Train the feature encoder using replay buffer."""
        if len(self.replay) < self.config.batch_size:
            return 0.0
        
        # Sampling: mixture of recent and old experiences
        replay_list = list(self.replay)
        mid = len(replay_list) // 2
        recent = replay_list[mid:]
        old = replay_list[:mid]
        
        recent_samples = np.random.choice(len(recent), self.config.batch_size // 2, replace=True)
        old_samples = np.random.choice(len(old), self.config.batch_size // 2, replace=True)
        
        batch = [recent[i] for i in recent_samples] + [old[i] for i in old_samples]
        
        feature_keys = ['mu', 'sigma', 'H', 'sharpness', 'sat_hi', 
                       'sat_lo', 'delta_mu', 'exp']
        
        raw_features_list = []
        for raw_feat, _, _ in batch:
            x = np.array([raw_feat[k] for k in feature_keys], dtype=np.float32)
            raw_features_list.append(x)
        
        features_tensor = torch.from_numpy(np.array(raw_features_list)).to(self.device)
        rewards = torch.tensor([r for _, _, r in batch], dtype=torch.float32).to(self.device)
        
        self.encoder.train()
        self.optimizer.zero_grad()
        
        _, reward_preds = self.encoder(features_tensor, return_reward=True)
        reward_preds = reward_preds.squeeze()
        
        loss = nn.MSELoss()(reward_preds, rewards)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.optimizer.step()
        
        self.encoder.eval()
        return loss.item()
    
    def recondition(self):
        """Regularization to prevent matrix singularity."""
        for action in self.actions:
            d = self.config.feature_dim
            identity = np.eye(d) / self.config.lambda_reg
            self.A_inv[action] = 0.9 * self.A_inv[action] + 0.1 * identity
    
    def add_actions(self, new_actions):
        for action in new_actions:
            if action not in self.actions:
                self.actions.append(action)
                self.init_action_params(action)


# ============================================================================
# 5. Hardware Interface (oCam-1MGN-U-T)
# ============================================================================

class OCamController:
    def __init__(self, device_id=2, fps=30, width=1280, height=720):
        self.device_id = device_id
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: /dev/video{device_id}")
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Manual Exposure Mode
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        
        # [NOTE] 30 FPS requires max exposure < 33ms (330 units)
        self.exp_range = (100, 330) 
        self.current_exp = 200
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.current_exp)
        
        time.sleep(0.3)
    
    def apply_exposure(self, exp):
        exp = int(np.clip(exp, self.exp_range[0], self.exp_range[1]))
        success = self.cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        if success:
            self.current_exp = exp
        return success
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    def release(self):
        self.cap.release()


# ============================================================================
# 6. Reward Function
# ============================================================================

class RewardCalculator:
    def __init__(self, a=1.0, b=0.3, 
                 c=0.01,      # [TUNED] Reduced brightness penalty (0.015 -> 0.01)
                 d=5.0, e=0.1, f=0.03, 
                 mu_target=95, # [TUNED] Lower target for night (115 -> 95)
                 clip_range=(-50, 10)):
        self.a = a  # Weight for Sharpness
        self.b = b  # Weight for Entropy
        self.c = c  # Weight for Brightness deviation penalty
        self.d = d  # Weight for Saturation penalty
        self.e = e  # Weight for Flicker penalty
        self.f = f  # Weight for Action cost penalty
        self.mu_target = mu_target
        self.clip_range = clip_range
        self.prev_mu = None
    
    def compute(self, norm_features, raw_features, action, current_exp, max_exp):
        sharp_n = norm_features.get('sharpness', 0)
        ent_n = norm_features.get('H', 0)
        
        mu = raw_features['mu']
        sat_hi = raw_features['sat_hi']
        sat_lo = raw_features['sat_lo']
        
        reward = 0.0
        
        # 1. Maximize Sharpness & Entropy (Image Information)
        reward += self.a * sharp_n
        reward += self.b * ent_n
        
        # 2. Penalties
        is_saturated_dark = (current_exp >= max_exp - 5) and (mu < self.mu_target)
        
        if not is_saturated_dark:
            reward -= self.c * (mu - self.mu_target) ** 2
            reward -= self.d * sat_lo
        else:
            # Compensation if hardware limited
            reward += 0.5


        reward -= self.d * sat_hi
        
        # 3. Flicker Penalty (Temporal Stability)
        if self.prev_mu is not None:
            reward -= self.e * abs(mu - self.prev_mu)
        self.prev_mu = mu
        
        # 4. Action Cost (Minimize rapid changes)
        delta_exp = action
        reward -= self.f * abs(delta_exp)
        
        # Clip reward to prevent gradient explosion
        reward = np.clip(reward, self.clip_range[0], self.clip_range[1])
        
        return reward


# ============================================================================
# 7. Main RL System & Loop
# ============================================================================

class OCamAutoExposureRL:
    def __init__(self, device_id=2, fps=30, width=1280, height=720, control_hz=15):
        self.camera = OCamController(device_id, fps, width, height)
        self.control_period = 1.0 / control_hz
        self.feature_extractor = FeatureExtractor()
        
        # [TUNED] Added larger steps for night mode adaptation (+-10)
        initial_actions = [-10, -3, -1, 0, 1, 3, 10]
        
        self.agent = NeuralLinUCB(input_dim=8, actions=initial_actions)
        self.reward_calculator = RewardCalculator()
        
        # Logging
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_filename = f'ocam_rl_log_{timestamp}.csv'
        self.log_file = open(log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'step', 'time', 'mu', 'sigma', 'entropy', 'sharpness',
            'sat_hi', 'sat_lo', 'exp_100us', 'exp_ms',
            'delta_exp', 'reward', 'mean_pred', 'uncertainty',
            'ucb_score', 'action_idx', 'num_actions'
        ])
        
        self.recent_rewards = deque(maxlen=200)
        self.recent_flicker = deque(maxlen=200)
        self.recent_blind = deque(maxlen=200)
        self.step = 0
        self.total_updates = 0
        self.expansion_stage = 0
        
        print(f"Log file: {log_filename}")

    def get_roi_frame(self, frame):
        # Region of Interest: Center 50%
        h, w = frame.shape[:2]
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
    
    def check_expansion_criteria(self):
        """Criteria for expanding action space (Curriculum Learning)."""
        if len(self.recent_rewards) < 200:
            return False
        avg_reward = np.mean(self.recent_rewards)
        avg_flicker = np.mean(self.recent_flicker)
        blind_rate = np.mean(self.recent_blind)
        
        if avg_reward > -4.0 and avg_flicker < 8.0 and blind_rate < 0.12:
            return True
        return False
    
    def expand_actions(self, stage):
        if stage == 1:
            new_actions = [-5, -2, 2, 5]
            self.agent.add_actions(new_actions)
            self.expansion_stage = 1
        elif stage == 2:
            new_actions = [-10, -7, 7, 10] # Already includes 10, but safe to keep
            self.agent.add_actions(new_actions)
            self.expansion_stage = 2
    
    def run_step(self):
        step_start = time.monotonic()
        
        # 1. Capture
        frame = self.camera.get_frame()
        if frame is None: return False
        roi, roi_coords = self.get_roi_frame(frame)
        
        # 2. Extract State
        norm_features, raw_features = self.feature_extractor.extract_features(roi, self.camera.current_exp)
        
        # 3. Select Action (LinUCB)
        action, phi, action_info, _ = self.agent.select_action(norm_features)
        
        # 4. Apply Action
        new_exp = self.camera.current_exp + action
        self.camera.apply_exposure(new_exp)
        
        # 5. Get Next State & Compute Reward
        eval_frame = self.camera.get_frame()
        if eval_frame is None: return False
        eval_roi, _ = self.get_roi_frame(eval_frame)
        eval_norm_features, eval_raw_features = self.feature_extractor.extract_features(eval_roi, self.camera.current_exp)
        
        reward = self.reward_calculator.compute(
        eval_norm_features, 
        eval_raw_features, 
        action, 
        self.camera.current_exp,  
        self.camera.exp_range[1]   
    )
            
        # 6. Update Agent
        self.agent.update(phi, action, reward, eval_norm_features)
        
        # 7. Encoder Training (Representation Learning)
        if self.step % self.agent.config.update_every == 0 and self.step > 0:
            loss = self.agent.train_encoder()
            self.total_updates += 1
        
        # 8. Reconditioning
        if self.step % 500 == 0 and self.step > 0:
            self.agent.recondition()
        
        # Logging & Visualization logic (omitted for brevity, keeping functional)
        self.recent_rewards.append(reward)
        self.recent_flicker.append(abs(eval_raw_features['delta_mu']))
        sat_total = eval_raw_features['sat_hi'] + eval_raw_features['sat_lo']
        self.recent_blind.append(1.0 if sat_total > 0.1 else 0.0)
        
        self.csv_writer.writerow([
            self.step, time.time(), eval_raw_features['mu'], eval_raw_features['sigma'],
            eval_raw_features['H'], eval_raw_features['sharpness'], eval_raw_features['sat_hi'],
            eval_raw_features['sat_lo'], self.camera.current_exp, self.camera.current_exp * 0.1,
            action, reward, action_info['mean'], action_info['unc'], action_info['score'],
            self.agent.actions.index(action), len(self.agent.actions)
        ])

        # Visualization
        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display_frame, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), (0, 255, 0), 2)
        
        # Info Overlay
        cv2.putText(display_frame, f"Exp: {self.camera.current_exp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Mu: {eval_raw_features['mu']:.1f} (T: {self.reward_calculator.mu_target})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('oCam Auto Exposure RL', display_frame)
        
        # Curriculum Learning Updates
        if self.step == 400 and self.expansion_stage == 0 and self.check_expansion_criteria():
            self.expand_actions(stage=1)
        elif self.step == 800 and self.expansion_stage == 1 and self.check_expansion_criteria():
            self.expand_actions(stage=2)
        
        # Reduce Exploration (Annealing Alpha)
        if self.step == 400: self.agent.config.alpha = 1.0
        elif self.step == 800: self.agent.config.alpha = 0.7
        
        self.step += 1
        
        elapsed = time.monotonic() - step_start
        if elapsed < self.control_period:
            time.sleep(self.control_period - elapsed)
        
        return True
    
    def run(self, max_steps=None):
        try:
            print("Starting RL Control Loop...")
            while True:
                if max_steps and self.step >= max_steps: break
                if not self.run_step(): break
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27: break  # ESC
                elif key == ord('s'): self.save_model()
                
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            self.cleanup()

    def save_model(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        torch.save(self.agent.encoder.state_dict(), f'ocam_encoder_{timestamp}.pth')
        np.savez(f'ocam_linucb_{timestamp}.npz',
            actions=self.agent.actions,
            theta={str(k): v for k, v in self.agent.theta.items()},
            A_inv={str(k): v for k, v in self.agent.A_inv.items()},
            b={str(k): v for k, v in self.agent.b.items()}
        )
        print("Model saved.")

    def cleanup(self):
        self.save_model()
        self.log_file.close()
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--hz', type=int, default=15)
    parser.add_argument('--steps', type=int, default=None)
    args = parser.parse_args()
    
    system = OCamAutoExposureRL(args.device, args.fps, args.width, args.height, args.hz)
    system.run(max_steps=args.steps)