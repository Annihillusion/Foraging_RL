import os
import numpy as np
import torch
import shelve
from datetime import datetime

from Environment import make_vec_envs
from params import parse_args
from train import train_one_episode
from visualize import test_and_plot
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr import algo, utils


def train(args):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(f'Using device: {device}')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, True, args.num_episode_steps, args.radius)

    # actor_critic = Policy(
    #     envs.observation_space.shape,
    #     envs.action_space,
    #     base_kwargs={'recurrent': args.recurrent_policy, 'hidden_size': args.hidden_size})
    # actor_critic.to(device)

    record_name = "2024-05-13 17-02"
    actor_critic, _ = torch.load(os.path.join(args.save_dir, record_name + ".pt"), map_location="cpu")

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_update_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    rollouts.to(device)

    pos_collector = []
    action_collector = []
    energy_collector = []
    reward_collector = []

    file_name = datetime.now().strftime('%Y-%m-%d %H-%M 6to12')
    # with shelve.open(os.path.join(args.log_dir, file_name + ' args.shelve')) as shelf:
    #     for arg in vars(args):
    #         shelf[arg] = getattr(args, arg)

    for episode in range(args.num_episodes):
        episode_action, episode_reward, _ = train_one_episode(envs, agent, rollouts, args, episode)
        episode_pos, _, episode_energy = test_and_plot(agent.actor_critic, args, file_name, episode, save_mode=True)

        pos_collector.append(episode_pos)
        action_collector.append(episode_action)
        energy_collector.append(episode_energy)
        reward_collector.append(episode_reward)

        positions = np.array(pos_collector)
        actions = np.array(action_collector)
        energies = np.array(energy_collector)
        rewards = np.array(reward_collector)

        torch.save([
            actor_critic,
            getattr(utils.get_vec_normalize(envs), 'obs_rms', None)],
            os.path.join(args.save_dir, file_name + '.pt'))
        np.savez(os.path.join(args.log_dir, file_name), position=positions, action=actions, energy=energies, reward=rewards)


if __name__ == '__main__':
    args = parse_args()
    args.radius = 12
    train(args)
