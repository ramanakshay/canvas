from agent.model import DiscreteActorCritic
from algorithm.policy_gradient import VanillaPolicyGradient
import gymnasium as gym


## HYPERPARAMETERS ##
LEARNING_RATE = 1e-3
EPOCHS = 5
HIDDEN_DIM = 64
TOTAL_TIMESTEPS = 150000
BATCH_TIMESTEPS = 2500
GAMMA = 0.99
LAM = 0.9

## ENVIRONMENT ##
env_config = {
    "max_ep_steps": 500
}
env = gym.make("Acrobot-v1", max_episode_steps = env_config["max_ep_steps"])

## MODEL ##
model_config = {
    "obs_dim": env.observation_space.shape[0],
    "act_dim": env.action_space.n,
    "hidden_dim": HIDDEN_DIM,
    "learning_rate": LEARNING_RATE
}
model = DiscreteActorCritic(model_config)

## ALGORITHM ##
alg_config = {
    "total_timesteps": TOTAL_TIMESTEPS,
    "timesteps_per_batch": BATCH_TIMESTEPS,
    "gamma": GAMMA,
    "lam": LAM

}
alg = VanillaPolicyGradient(model, env, alg_config)
alg.run()



