import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=202434,
                        help='random seed (default: 202434)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=16,
                        help='number of batches for ppo')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--recurrent-policy',action='store_true',default=True,
                        help='use a recurrent policy')
    parser.add_argument('--env-name', default='CircularEnv',
                        help='environment to train on (default: CircularEnv)')
    parser.add_argument('--log-dir', default='logs',
                        help='directory to save agent logs (default: logs)')
    parser.add_argument('--save-dir', default='./trained_models',
                        help='directory to save agent logs (default: trained_models)')
    parser.add_argument('--num-update-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--num-epoch-steps', type=int, default=10e2,
                        help='number of environment steps to train in single epoch')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=4)
    parser.add_argument("--render", type=str, default=None)

    args = parser.parse_args()
    return args