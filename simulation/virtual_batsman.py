import numpy as np
from typing import Dict, Tuple

class VirtualBatsman:
    def __init__(self, skill_level: float = 0.7, weakness: str = "short_balls"):
        self.skill = skill_level
        self.weakness = weakness
        self.shot_history = []
        self.confidence = 0.5
        
    def play_shot(self, ball_params: Dict) -> Tuple[str, float, bool]:
        """
        Determine batsman's response to a delivery
        Returns: (shot_type, timing_quality, is_successful)
        """
        line = ball_params['final_line']
        length = ball_params['length']
        speed = ball_params['speed']
        
        # Determine appropriate shot based on line/length
        shot_type = self._select_shot(line, length)
        
        # Calculate success probability
        success_prob = self._calculate_success_probability(shot_type, ball_params)
        
        # Add some randomness based on skill
        timing_quality = np.random.normal(self.skill, 0.2)
        timing_quality = np.clip(timing_quality, 0.1, 1.0)
        
        # Determine if shot is successful
        is_successful = (timing_quality * success_prob) > 0.5
        
        # Update shot history
        self.shot_history.append((shot_type, is_successful))
        if len(self.shot_history) > 10:
            self.shot_history.pop(0)
            
        # Update confidence based on recent performance
        self._update_confidence()
        
        return shot_type, timing_quality, is_successful
    
    def _select_shot(self, line: float, length: float) -> str:
        """Select appropriate shot based on ball parameters"""
        if length < 6:  # Yorker
            return "defense" if abs(line) < 0.3 else "leave"
        elif length < 9:  # Full length
            if abs(line) < 0.2:
                return "drive"
            else:
                return "cover_drive" if line > 0 else "on_drive"
        elif length < 12:  # Good length
            return "defense" if abs(line) < 0.3 else "cut" if line > 0.2 else "pull"
        else:  # Short ball
            return "pull" if abs(line) < 0.4 else "leave"
    
    def _calculate_success_probability(self, shot_type: str, ball_params: Dict) -> float:
        """Calculate probability of successful shot execution"""
        base_probabilities = {
            "drive": 0.7, "cover_drive": 0.6, "on_drive": 0.6,
            "cut": 0.8, "pull": 0.5, "defense": 0.9, "leave": 1.0
        }
        
        base_prob = base_probabilities.get(shot_type, 0.5)
        
        # Adjust for weaknesses
        if self.weakness == "short_balls" and ball_params['length'] > 10:
            base_prob *= 0.7
        elif self.weakness == "spin" and ball_params['spin_magnitude'] > 25:
            base_prob *= 0.8
            
        # Adjust for speed (very fast balls are harder to play)
        if ball_params['speed'] > 45:  # > 162 km/h
            base_prob *= 0.9
            
        return base_prob * self.confidence
    
    def _update_confidence(self):
        """Update batsman confidence based on recent performance"""
        if len(self.shot_history) == 0:
            return
            
        recent_success_rate = sum(1 for _, success in self.shot_history[-5:] if success) / 5
        self.confidence = 0.3 + recent_success_rate * 0.7
        
    def get_state(self) -> np.ndarray:
        """Get batsman's current state as feature vector"""
        recent_success = sum(1 for _, success in self.shot_history[-3:] if success) / 3 if self.shot_history else 0.5
        
        return np.array([
            self.skill,
            self.confidence,
            recent_success,
            1.0 if self.weakness == "short_balls" else 0.0,
            1.0 if self.weakness == "spin" else 0.0
        ])