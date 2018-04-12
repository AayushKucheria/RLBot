import argparse
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import os
import sys
import pickle
import time
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.policy import Policy
from models.critic import Value
from torch.autograd import Variable
from core.a2c import a2c_step
from core.common import estimate_advantages
from core.agent import Agent

parser = argparse.ArgumentParser(description='PyTorch A2C example')
parser.add_argument('--env', required=True,
                    help='name of the environment to run')
parser.add_argument('--model-path',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--num-threads', type=int, default=4,
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048,
                    help='minimal batch size per A2C update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500,
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1,
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=0,
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--discount', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='learning rate (default: 7e-4)')
parser.add_argument('--tau', type=float, default=1,
                    help='gae parameter (default: 1)')
args = parser.parse_args()


def env_factory(thread_id):
    env = gym.make(args.env)
    env.seed(args.seed + thread_id)
    env = FlatObsWrapper(env)
    return env


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

env_dummy = env_factory(0)

"""define actor and critic"""
if args.model_path is None:
    policy_net = Policy(env_dummy.observation_space, env_dummy.action_space)
    value_net = Value(env_dummy.observation_space)
else:
    policy_net, value_net = pickle.load(open(args.model_path, "rb"))
if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()
del env_dummy

policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)

"""create agent"""
agent = Agent(env_factory, policy_net, render=args.render, num_threads=args.num_threads)


def update_params(batch):
    obss = torch.from_numpy(np.stack(batch.obs)).float()
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward)).float()
    masks = torch.from_numpy(np.stack(batch.mask)).float()
    if use_gpu:
        obss, actions, rewards, masks = obss.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
    values = value_net(Variable(obss, volatile=True)).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.discount, args.tau, use_gpu)

    """perform A2C update"""
    a2c_step(policy_net, value_net, policy_optimizer, value_optimizer, obss, actions, returns, advantages)


def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            if use_gpu:
                policy_net.cpu(), value_net.cpu()
            pickle.dump((policy_net, value_net),
                        open(os.path.join(assets_dir(), 'learned_models/{}_a2c.p'.format(args.env)), 'wb'))
            if use_gpu:
                policy_net.cuda(), value_net.cuda()


main_loop()
