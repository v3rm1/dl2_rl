dic_agent_conf = {
    "STATE_DIM": (80, 80, 1),
    "ACTOR_LEARNING_RATE": 1e-5,
    "CRITIC_LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 10,
    "GAMMA": 0.99,
    "PATIENCE": 10,
    "NUM_LAYERS": 2,
    "D_DENSE": 64,
    "ACTOR_LOSS": "Clipped",  # or "KL-DIVERGENCE"
    "CLIPPING_LOSS_RATIO": 0.1,
    "ENTROPY_LOSS_RATIO": 0.1,
    "CRITIC_LOSS": "mean_squared_error",
    "OPTIMIZER": "Adam",
    "TARGET_UPDATE_ALPHA": 0.9,
}

dic_env_conf = {
    "ENV_NAME": "PongNoFrameskip-v4",
    "GYM_SEED": 1,
    "LIST_STATE_NAME": ["state"],
    "ACTION_RANGE": "-1-1", # or "-1~1"
    "POSITIVE_REWARD": False
}

dic_path ={
    "PPO": "records/PPO/"
}

dic_exp_conf = {
    "TRAIN_ITERATIONS": 1000,
    "MAX_EPISODE_LENGTH": 1000,
    "TEST_ITERATIONS": 10
}

if __name__ == "__main__":
    import gym

    env = gym.make(dic_env_conf["ENV_NAME"])

    print(env.action_space.n)
    print(env.reward_range)
    print(env.observation_space)
