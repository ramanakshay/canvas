from agent.model import DiscreteActorCritic
from algorithm.policy_gradient import VanillaPolicyGradient

import gymnasium as gym

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## ENVIRONMENT ##
    env = gym.make(config.env.name, max_episode_steps = config.env.max_ep_steps)

    ## MODEL ##
    model = DiscreteActorCritic(config.agent)

    ## ALGORITHM ##
    alg = VanillaPolicyGradient(env, model, config.algorithm)
    alg.run()

if __name__ == "__main__":
    main()