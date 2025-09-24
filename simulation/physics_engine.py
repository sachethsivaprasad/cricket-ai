import numpy as np
from typing import Tuple, Dict

class CricketPhysicsEngine:
    def __init__(self):
        self.gravity = 9.8
        self.pitch_length = 20.0  # meters from bowler to batsman
        self.drag_coefficient = 0.45
        self.air_density = 1.2
        self.ball_radius = 0.036  # meters
        self.ball_mass = 0.16     # kg
        
    def simulate_delivery(self, action: np.ndarray) -> Dict:
        """
        Simulate a ball delivery based on action parameters
        Action: [speed_normalized, line_normalized, length_normalized, spin_type, spin_magnitude]
        """
        # Denormalize parameters
        speed = 20 + action[0] * 40  # 20-60 m/s (72-216 km/h)
        line = -0.5 + action[1]  # -0.5 to +0.5 (off to leg side)
        length = 4 + action[2] * 10  # 4-14 meters from batsman (yorker to bouncer)
        spin_type = action[3]  # 0: offspin, 1: legspin
        spin_magnitude = action[4] * 50  # 0-50 rpm
        
        # Calculate trajectory (simplified projectile motion with drag)
        flight_time = self._calculate_flight_time(speed, length)
        
        # Apply spin effects
        lateral_movement = self._calculate_spin_movement(spin_type, spin_magnitude, flight_time)
        final_line = line + lateral_movement
        
        # Calculate bounce
        bounce_deviation = self._calculate_bounce(spin_magnitude, spin_type)
        
        return {
            'speed': speed,
            'initial_line': line,
            'length': length,
            'final_line': final_line,
            'flight_time': flight_time,
            'bounce_deviation': bounce_deviation,
            'spin_type': spin_type,
            'spin_magnitude': spin_magnitude
        }
    
    def _calculate_flight_time(self, speed: float, length: float) -> float:
        """Calculate time taken for ball to reach batsman"""
        # Simplified - ignoring air resistance for time calculation
        return length / speed
    
    def _calculate_spin_movement(self, spin_type: float, spin_magnitude: float, flight_time: float) -> float:
        """Calculate lateral movement due to spin"""
        # Spin type: 0 = offspin (moves away from right-handed batsman)
        #            1 = legspin (moves into right-handed batsman)
        direction = -1 if spin_type < 0.5 else 1
        movement = spin_magnitude * flight_time * 0.001  # Simplified model
        return direction * movement
    
    def _calculate_bounce(self, spin_magnitude: float, spin_type: float) -> float:
        """Calculate deviation after bounce"""
        # More spin = more deviation
        bounce = spin_magnitude * 0.002
        return bounce

# Example usage
if __name__ == "__main__":
    physics = CricketPhysicsEngine()
    test_action = np.array([0.5, 0.5, 0.5, 0.0, 0.5])  # Medium pace, middle stump, good length
    result = physics.simulate_delivery(test_action)
    print("Ball delivery result:", result)