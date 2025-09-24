import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Optional

from simulation.physics_engine import CricketPhysicsEngine
from simulation.virtual_batsman import VirtualBatsman

class CricketBowlEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        super(CricketBowlEnv, self).__init__()
        
        # Initialize components
        self.physics_engine = CricketPhysicsEngine()
        self.batsman = VirtualBatsman(skill_level=0.7, weakness="short_balls")
        
        # Define action space: [speed, line, length, spin_type, spin_magnitude]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space: batsman_state + match_context
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 60.0, 36.0, 6.0]),
            dtype=np.float32
        )
        
        # Match state
        self.balls_bowled = 0
        self.runs_conceded = 0
        self.wickets_taken = 0
        self.current_over_balls = 0
        self.runs_last_over = 0
        
        # Gym 0.26+ requires render_mode
        self.render_mode = render_mode
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.balls_bowled = 0
        self.runs_conceded = 0
        self.wickets_taken = 0
        self.current_over_balls = 0
        self.runs_last_over = 0
        
        # Reset batsman
        self.batsman = VirtualBatsman(
            skill_level=self.np_random.uniform(0.5, 0.9),
            weakness=self.np_random.choice(["short_balls", "spin", "pace"])
        )
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step (one ball)"""
        # Simulate ball delivery
        ball_params = self.physics_engine.simulate_delivery(action)
        
        # Batsman plays shot
        shot_type, timing_quality, is_successful = self.batsman.play_shot(ball_params)
        
        # Calculate reward
        reward = self._calculate_reward(ball_params, shot_type, is_successful, timing_quality)
        
        # Update match state
        self.balls_bowled += 1
        self.current_over_balls += 1
        
        if is_successful:
            runs_scored = self._calculate_runs_scored(shot_type, timing_quality)
            self.runs_conceded += runs_scored
            if self.current_over_balls <= 6:
                self.runs_last_over += runs_scored
        else:
            # Chance of wicket when shot unsuccessful
            if timing_quality < 0.3 and self.np_random.random() < 0.3:
                self.wickets_taken += 1
                reward += 20  # Big reward for taking wicket
        
        # Check if over is complete
        if self.current_over_balls >= 6:
            self.current_over_balls = 0
            self.runs_last_over = 0
        
        # Check if innings complete (simplified)
        terminated = self.balls_bowled >= 60 or self.wickets_taken >= 3
        truncated = False  # We don't use time limits
        
        # Additional info
        info = {
            'ball_params': ball_params,
            'shot_type': shot_type,
            'successful': is_successful,
            'runs_conceded': self.runs_conceded,
            'wickets': self.wickets_taken
        }
        
        observation = self._get_observation()
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, ball_params: Dict, shot_type: str, 
                         is_successful: bool, timing_quality: float) -> float:
        """Calculate reward for the bowler's action"""
        reward = 0.0
        
        if not is_successful:
            # Batsman missed or played poorly
            reward += 10.0
            if timing_quality < 0.3:
                reward += 5.0  # Extra reward for beating bat completely
        else:
            # Batsman scored runs
            runs = self._calculate_runs_scored(shot_type, timing_quality)
            reward -= runs * 2.0  # Penalize for runs conceded
            
        # Bonus for good line/length
        length = ball_params['length']
        if 7.5 < length < 9.5:  # Good length area
            reward += 2.0
            
        # Economy rate bonus/penalty
        if self.balls_bowled > 0:
            economy_rate = self.runs_conceded / (self.balls_bowled / 6)
            if economy_rate < 6.0:  # Good economy
                reward += 1.0
            elif economy_rate > 9.0:  # Poor economy
                reward -= 1.0
            
        return reward
    
    def _calculate_runs_scored(self, shot_type: str, timing_quality: float) -> int:
        """Calculate runs scored from a successful shot"""
        base_runs = {
            "drive": 4, "cover_drive": 4, "on_drive": 3,
            "cut": 4, "pull": 6, "defense": 0, "leave": 0
        }
        
        runs = base_runs.get(shot_type, 1)
        
        # Adjust based on timing quality
        if timing_quality > 0.8 and runs > 0:
            runs += 1  # Well-timed shot gets extra runs
            
        return runs
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation of the environment"""
        batsman_state = self.batsman.get_state()
        
        # Match context
        balls_remaining = 60 - self.balls_bowled
        over_progress = self.current_over_balls
        
        observation = np.concatenate([
            batsman_state,
            np.array([balls_remaining, self.runs_last_over, over_progress])
        ])
        
        return observation.astype(np.float32)
    
    def render(self):
        """Simple text rendering"""
        print(f"Balls: {self.balls_bowled} | Runs: {self.runs_conceded}/{self.wickets_taken} | "
              f"Over: {self.current_over_balls}/6 | Batsman Confidence: {self.batsman.confidence:.2f}")
    
    def close(self):
        """Clean up resources"""
        pass