import argparse
from two_step.two_step import Agents
import Environment


def train(env, agent, num_episodes, max_steps, render):
    for episode in range(num_episodes):
        state = env.reset()
        action = agent.choose_action(state)

        for step in range(max_steps):
            next_state, reward, done, _ = agent.env.step(action)
            next_action = agent.choose_action(next_state)

            # 更新Q值
            agent.update(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action
            if render is not None:
                env.render()
            if done:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", default=0.7, type=float)  # learning rate
    parser.add_argument("--gamma", default=0.99, type=float)  # discount factor
    parser.add_argument("--epsilon", default=0.1, type=float)
    parser.add_argument("--train_episodes", default=100, type=int)
    parser.add_argument("--test_episodes", default=100, type=int)
    parser.add_argument("--max_steps", default=1000, type=int)
    parser.add_argument("--render", default='human', type=str)
    args = parser.parse_args()

    env = Environment.CircularEnv(render_mode=args.render)
    agent = Agents.SARSA(env, args.alpha, args.gamma, args.epsilon)
    train(env, agent, args.train_episodes, args.max_steps, args.render)
