import os
import torch
import torch.nn as nn
import copy
from Environment import make_vec_envs, CircularEnv
from params import parse_args
from a2c_ppo_acktr.utils import get_render_func
from a2c_ppo_acktr.model import Policy


def draw_trace(actor_critic, args):
    device = torch.device("cpu")
    env = make_vec_envs(args.env_name, args.seed, 1,
                        args.gamma, args.log_dir, device, True, args.num_episode_steps)
    # env = CircularEnv(5.0, step_size=0.1, render_mode=None)
    render_func = get_render_func(env)

    obs = env.reset()
    recurrent_hidden_states = torch.randn(1, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.ones(1, 1, device=device)

    eval_hidden_states = torch.randn(1, actor_critic.recurrent_hidden_state_size, device=device)
    # rnn = copy.deepcopy(actor_critic.base.rnn)
    rnn = nn.RNN(2, 16)
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)

    for i in range(1000):
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = actor_critic.act(
                obs,
                recurrent_hidden_states,
                masks,
                deterministic=True)

            # outputs, eval_hidden_states = rnn(obs, eval_hidden_states)

        # Obser reward and next obs
        obs, _, done, infos = env.step(action)
    render_func()
    env.close()


if __name__ == '__main__':
    args = parse_args()
    actor_critic, obs_rms = torch.load(os.path.join(args.save_dir, "2024-04-30 18-37.pt"), map_location="cpu")
    draw_trace(actor_critic, args)
