import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
from utils import get_vec_normalize
from visualize import visdom_plot

from bw_module_mujoco import bw_module

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        vizz = Visdom(port=args.port)
        win = None
        winloss = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)
    #Initialize bw Model
    if args.bw:
        bw_model = bw_module(actor_critic, args, agent.optimizer, envs.action_space, envs.observation_space)
    vis_timesteps = []
    vis_loss = []
    
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

            # Add stuff to the Buffer
            if args.bw:
                bw_model.step(rollouts.obs[step].detach().cpu().numpy(),
                            action.detach().cpu().numpy(),
                            reward.detach().cpu().numpy(),
                            done,
                            obs.detach().cpu().numpy())
        
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # Do BW STEPS
        if args.bw and (j % args.n_a2c == 0):
            if not args.consistency:
                l_bw, l_imi = 0.0, 0.0
                for _ in range(args.n_bw):
                    l_bw += bw_model.train_bw_model(j)
                l_bw /= args.n_bw
                for _ in range(args.n_imi):
                    l_imi += bw_model.train_imitation(j)
                l_imi /= args.n_imi
            else:
                l_bw, l_fw = 0.0, 0.0
                for _ in range(args.n_bw):
                    l_bw_, l_fw_ = bw_model.train_bw_model(j)
                    l_bw += l_bw_
                    l_fw += l_fw_
                l_bw /= args.n_bw
                l_fw /= args.n_bw
                l_imi, l_cons = 0.0, 0.0
                for _ in range(args.n_imi):
                    l_imi_, l_cons_ = bw_model.train_imitation(j)
                    l_imi += l_imi_
                    l_cons_ += l_cons_
                l_imi /= args.n_imi
                l_cons /= args.n_imi

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                env_name = args.env_name
                if args.bw:
                    env_name += 'BW'
                win = visdom_plot(viz, win, args.log_dir, env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass

        # Save to Visdom Plots
        if args.vis and (j % args.vis_interval == 0):
            if args.bw and args.consistency:
                vis_loss.append([value_loss, action_loss, l_bw, l_imi, l_fw, l_cons])
                legend=['Value loss','Action loss', 'BW Loss','IMI loss','FW Loss', 'CONST loss']
                title = args.env_name + '-' + 'bw' + '-' + 'consistency' + args.title
            elif args.bw:
                vis_loss.append([value_loss, action_loss, l_bw, l_imi])
                legend=['Value loss','Action loss', 'BW Loss','IMI loss']
                title = args.env_name + '-' + 'bw' + args.title
            else:
                vis_loss.append([value_loss, action_loss])
                legend=['Value loss','Action loss']
                title = args.env_name + '-' + 'vanilla'
            vis_timesteps.append((j+1)*(args.num_processes * args.num_steps))
            # vis_rewards.append(final_rewards.mean())
            # vis_rewards.append(np.mean(reward_queue))
            
            # if win is None:
            #     win = vizz.line(Y=np.array(vis_rewards), X=np.array(vis_timesteps), opts=dict(title=title, xlabel='Timesteps',
            #                 ylabel='Avg Rewards'))
            # vizz.line(Y=np.array(vis_rewards), X=np.array(vis_timesteps), win=win, update='replace', opts=dict(title=title, xlabel='Timesteps',
            #                 ylabel='Avg Rewards'))
            if winloss is None:
                winloss = vizz.line(Y=np.array(vis_loss), X=np.array(vis_timesteps), opts=dict(title=title, xlabel='Timesteps',
                            ylabel='Losses', legend=legend))
            vizz.line(Y=np.array(vis_loss), X=np.array(vis_timesteps), win=winloss, update='replace', opts=dict(title=title, xlabel='Timesteps',
                            ylabel='Losses', legend=legend))

if __name__ == "__main__":
    main()
