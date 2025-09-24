import gym
from stable_baselines3 import PPO
from simulation.cricket_env import CricketBowlEnv

def evaluate_agent(model_path: str = "cricket_bowling_ai", num_episodes: int = 10):
    """Evaluate the trained agent"""
    
    # Load environment and model
    env = CricketBowlEnv(render_mode="human")
    model = PPO.load(model_path)
    
    total_rewards = []
    wickets_taken = []
    runs_conceded = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        print(f"\n=== Episode {episode + 1} ===")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Render the environment
            env.render()
            
            if done:
                print(f"Episode finished: "
                      f"Runs: {info[0]['runs_conceded']} | "
                      f"Wickets: {info[0]['wickets']} | "
                      f"Total Reward: {episode_reward:.2f}")
                
                total_rewards.append(episode_reward)
                wickets_taken.append(info[0]['wickets'])
                runs_conceded.append(info[0]['runs_conceded'])
    
    # Print summary statistics
    print(f"\n=== Evaluation Summary ===")
    print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Average Wickets: {sum(wickets_taken)/len(wickets_taken):.2f}")
    print(f"Average Runs Conceded: {sum(runs_conceded)/len(runs_conceded):.2f}")
    
    env.close()

if __name__ == "__main__":
    evaluate_agent()