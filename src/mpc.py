import os
import numpy as np
import gym
import torch
from dynamics_model import FC

def shooting(env, model, x0, horizon, num_sample):
    max_reward = -1e9
    action = env.action_space.sample()
    for i in range(num_sample):
        actions = [env.action_space.sample() for _ in range(horizon)]
        pred_reward = model_rollout(env, model, x0, actions)
        if pred_reward > max_reward:
            max_reward = pred_reward
            action = actions[0]
    return action

def model_rollout(env, model, x0, actions):
    x = x0
    expected_reward = 0
    for a in actions:
        dx = model.forward(torch.tensor(np.hstack([x, a])).float()).detach().numpy()
        expected_reward += env.predict_reward(x, a, x + dx)
        x = x + dx

    return expected_reward


def simulateion(env, model, render = False, use_shooting = False):
    horizon = 5
    num_sample = 100
    
    state = env.reset()
    done = False

    total_reward = 0
    max_steps = 100
    for i in range(max_steps):        
        action = env.action_space.sample()
        if use_shooting:
            action = shooting(env, model, state, horizon, num_sample)
        
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break
        if render:
            env.render()
            
    
    return total_reward

def evaluate(args):
    env = gym.make(args["env_name"])
    model = FC(args["num_layer"], args["input_dim"], args["hidden_dim"], args["output_dim"])
    model.load_state_dict(torch.load(args["load_path"]))
    print("Random action:")
    print(np.mean([simulateion(env, model, use_shooting=False, render=True) for i in range(1)]))
    print("Shooting method:")
    print(np.mean([simulateion(env, model, use_shooting=True, render=True) for i in range(1)]))


if __name__ == "__main__":

    eval_ant_args = {
        "env_name": "gym_free_ant:Free-Ant-v0",
        "num_layer": 2,
        "input_dim": 121,
        "hidden_dim": 500,
        "output_dim": 113,
        "load_path": "../model/ant-FC2-100/epoch_200.pth",
    }
    evaluate(eval_ant_args)

