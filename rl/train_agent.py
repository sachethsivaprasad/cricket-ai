import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from simulation.cricket_env import CricketBowlEnv

def load_config():
    with open('config/parameters.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_env():
    """Create and wrap the environment"""
    env = CricketBowlEnv()
    env = DummyVecEnv([lambda: env])
    return env

def train_agent():
    """Main training function"""
    config = load_config()
    
    # Create environment
    env = create_env()
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="cricket_ppo"
    )
    
    # Start training
    print("Starting training...")
    model.learn(
        total_timesteps=100000,
        callback=checkpoint_callback,
        tb_log_name="first_run"
    )
    
    # Save the final model
    model.save("cricket_bowling_ai")
    print("Training completed. Model saved.")
    
    env.close()

if __name__ == "__main__":
    train_agent()