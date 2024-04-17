import torch
import os
from tqdm import tqdm

from Environment import make_vec_envs
from params import parse_args
from train import train_one_epoch
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr import algo, utils


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, True)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

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

    for epoch in range(args.num_epochs):
        train_one_epoch(envs, agent, rollouts, args, epoch)
        torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(args.save_dir, 'ppo.pt'))


if __name__ == '__main__':
    main()
