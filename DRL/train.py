from collections import deque

import torch
import numpy as np
from tqdm import tqdm

from a2c_ppo_acktr import algo, utils


def train_one_episode(envs, agent, rollouts, args, episode_index):
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)

    episode_rewards = deque(maxlen=10)
    # num_updates = int(args.num_episode_steps) // args.num_update_steps // args.num_processes
    num_updates = int(args.num_episode_steps) // args.num_update_steps
    loss_record = np.empty([num_updates, 3])

    action_collector = []
    reward_collector = []

    for j in tqdm(range(num_updates), desc=f"Episode {episode_index + 1}/{args.num_episodes}"):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_update_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = agent.actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            action_collector.append(action.view(-1))
            reward_collector.append(reward.view(-1))

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = agent.actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, False)
        # value_loss, action_loss, dist_entropy
        loss_record[j] = agent.update(rollouts)
        rollouts.after_update()

    action_collector = np.array(action_collector)
    reward_collector = np.array(reward_collector)
    return action_collector, reward_collector, loss_record


        # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0
        #         or j == num_updates - 1) and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass
        #
        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
        #     ], os.path.join(save_path, args.env_name + ".pt"))
        #
        # if j % args.log_interval == 0 and len(episode_rewards) > 1:
        #     total_num_steps = (j + 1) * args.num_processes * args.num_steps
        #     end = time.time()
        #     print(
        #         "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
        #         .format(j, total_num_steps,
        #                 int(total_num_steps / (end - start)),
        #                 len(episode_rewards), np.mean(episode_rewards),
        #                 np.median(episode_rewards), np.min(episode_rewards),
        #                 np.max(episode_rewards), dist_entropy, value_loss,
        #                 action_loss))
        #
        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     obs_rms = utils.get_vec_normalize(envs).obs_rms
            # evaluate(actor_critic, obs_rms, args.env_name, args.seed,
            #          args.num_processes, eval_log_dir, device)
