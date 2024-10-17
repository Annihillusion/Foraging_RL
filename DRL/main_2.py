import os
import numpy as np
import torch
from datetime import datetime

from Environment import make_vec_envs
from params import parse_args
from train import train_one_episode
from visualize import test_and_plot
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr import algo, utils


def main(args):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    if args.exp_name is not None:
        os.makedirs(os.path.join('exp_records', args.exp_name), exist_ok=True)
        os.makedirs(os.path.join('exp_models', args.exp_name), exist_ok=True)
        os.makedirs(os.path.join('exp_plots', args.exp_name), exist_ok=True)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, True, args.num_episode_steps, args.radius)

    # “培育”阶段与“转移”阶段的模型参数设置
    if args.mode == 'train':
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy, 'hidden_size': args.hidden_size})
        actor_critic.to(device)
    elif args.mode == 'transfer':
        actor_critic, _ = torch.load(os.path.join('exp_models', args.exp_name, args.model_name + ".pt"), map_location="cpu")

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

    # file_name = datetime.now().strftime('%Y-%m-%d %H-%M')
    file_name = args.file_name

    # file_name = '2024-05-15 15-12'
    # with shelve.open(os.path.join(args.log_dir, file_name + ' args.shelve')) as shelf:
    #     for arg in vars(args):
    #         shelf[arg] = getattr(args, arg)

    pos, action, action_prob, actor_feature, rnn_x, energy, reward, _ \
        = train_one_episode(envs, agent, rollouts, args)
    # test_pos, _, test_energy = test_and_plot(agent.actor_critic, args, file_name, episode, save_mode=True)

    # 保存模型参数
    # torch.save([
    #     actor_critic,
    #     getattr(utils.get_vec_normalize(envs), 'obs_rms', None)],
    #     os.path.join('exp_models', args.exp_name, file_name + '.pt'))
    # 保存实验过程记录
    np.savez(os.path.join('exp_records', args.exp_name, 'small_'+file_name), position=pos, action=action,
             action_prob=action_prob, actor_feature=actor_feature, rnn_x=rnn_x, energy=energy, reward=reward)

    args.num_episode_steps = 20000
    args.lr = 4e-4
    args.radius = 12
    pos, action, action_prob, actor_feature, rnn_x, energy, reward, _ \
        = train_one_episode(envs, agent, rollouts, args)

    np.savez(os.path.join('exp_records', args.exp_name, 'small_large_'+file_name), position=pos, action=action,
             action_prob=action_prob, actor_feature=actor_feature, rnn_x=rnn_x, energy=energy, reward=reward)

if __name__ == '__main__':
    args = parse_args()
    main(args)
