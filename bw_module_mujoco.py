import numpy as np
import ipdb
import torch
import torch.nn as nn
import numpy as np
import random
from model import Policy, StateGen
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from utils_bw import select_mj, evaluate_mj, zero_mean_unit_std

class ReplayBuffer:
    """
    Replay buffer...
    """
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, R, obs_next):
        data = (obs_t, action, R, obs_next)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):
        obses_t, actions, returns, obses_next = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, R, obs_next = data
            obses_t.append(obs_t)
            actions.append(action)
            returns.append(R)
            obses_next.append(obs_next)
        return np.array(obses_t), np.array(actions), np.array(returns), np.array(obses_next)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 1e-6)
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size, beta):
        idxes = self._sample_proportional(batch_size)
        if beta > 0:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            weights = np.ones_like(idxes, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

class bw_module:
    def __init__(self, actor_critic, args, optimizer, action_shape, obs_shape):
        self.args = args
        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.action_shape = action_shape
        self.obs_shape = obs_shape
        #Create Backward models
        # self.bw_actgen = ActGen(self.obs_shape, self.action_shape)
        kwargs = {'hidden_size' : 128}
        self.bw_actgen = Policy(obs_shape.shape, action_shape, kwargs)
        # self.bw_stategen = StateGen(self.obs_shape, self.action_shape)
        self.bw_stategen = StateGen(obs_shape.shape, action_shape, kwargs)
        if self.args.cuda:
            self.bw_actgen.cuda()
            self.bw_stategen.cuda()
        self.bw_params = list(self.bw_actgen.parameters()) + list(self.bw_stategen.parameters())
        self.bw_optimizer = torch.optim.RMSprop(self.bw_params, lr=1e-3, eps=self.args.eps, alpha=self.args.alpha)
        # self.bw_optimizer = torch.optim.Adam(self.bw_params, lr=1e-3)
        
        #Create a forward model
        if self.args.consistency:
            self.fw_stategen = StateGen(obs_shape.shape, action_shape, kwargs)
            if self.args.cuda:
                self.fw_stategen.cuda()
            self.fw_optimizer = torch.optim.RMSprop(self.fw_stategen.parameters(), lr=self.args.lr, eps=self.args.eps, alpha=self.args.alpha)
            # self.fw_optimizer = torch.optim.Adam(self.fw_stategen.parameters(), lr=self.args.lr)
        #Create an episode buffer of size : # processes
        self.running_episodes = [[] for _ in range(self.args.num_processes)]
        if self.args.per_weight:
            self.buffer = PrioritizedReplayBuffer(self.args.capacity, self.args.sil_alpha)
        else:
            self.buffer = ReplayBuffer(self.args.capacity)
        # some other parameters...
        self.total_steps = []
        self.total_rewards = []
        # Set the mean, stds. All numpy stuff
        self.obs_delta_mean = None
        self.obs_delta_std = None
        self.obs_next_mean = None
        self.obs_next_std = None
        self.obs_mean = None
        self.obs_std = None
        self.actions_mean = None
        self.actions_std = None

    def train_bw_model(self, update):
        """
        Train the bw_model. Sample (s,a,r,s) from PER Buffer, Compute bw_model loss & Optimize
        """
        if self.args.per_weight:
            obs, actions, _, obs_next_unnormalized, weights, idxes = self.sample_batch(self.args.k_states)
        else:
            obs, actions, returns, obs_next_unnormalized = self.sample_batch_noper(self.args.capacity)
        batch_size = min(self.args.k_states, len(self.buffer))
        if obs is not None and obs_next_unnormalized is not None:
            if not self.args.per_weight:
                with torch.no_grad():
                    # self.actor_critic requires un-normalized states
                    obs_next_unnormalized = torch.tensor(obs_next_unnormalized, dtype=torch.float32)
                    if self.args.cuda:
                        obs_next_unnormalized = obs_next_unnormalized.cuda()
                    value = self.actor_critic.get_value(obs_next_unnormalized, None, None)
                    sorted_indices = value.cpu().numpy().reshape(-1).argsort()[-batch_size:][::-1]
                    obs_next_unnormalized = obs_next_unnormalized.cpu().numpy()
                # Select high value states under currect valuation for target states_next
                obs = obs[sorted_indices.tolist()]
                obs_next_unnormalized = obs_next_unnormalized[sorted_indices.tolist()]
                actions = actions[sorted_indices.tolist()]
                returns = returns[sorted_indices.tolist()]

            obs_delta, self.obs_delta_mean, self.obs_delta_std = zero_mean_unit_std(obs-obs_next_unnormalized)
            actions, self.actions_mean, self.actions_std = zero_mean_unit_std(actions)
            obs, self.obs_mean, self.obs_std = zero_mean_unit_std(obs)
            obs_next, self.obs_next_mean, self.obs_next_std = zero_mean_unit_std(obs_next_unnormalized)
            # need to get the masks
            # get basic information of network..
            obs_delta = torch.tensor(obs_delta, dtype=torch.float32)
            obs_next = torch.tensor(obs_next, dtype=torch.float32)
            obs = torch.tensor(obs, dtype=torch.float32)
            obs_next_unnormalized = torch.tensor(obs_next_unnormalized, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            if self.args.per_weight:
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
            if self.args.cuda:
                obs_delta = obs_delta.cuda()
                obs = obs.cuda()
                obs_next = obs_next.cuda()
                obs_next_unnormalized = obs_next_unnormalized.cuda()
                actions = actions.cuda()
                if self.args.per_weight:
                    weights = weights.cuda()
            # Train BW - Model
            avg_loss = 0
            for _ in range(self.args.epoch):
                # a_mu = self.bw_actgen(obs_next)
                _, action_log_probs, action_entropy, _ = self.bw_actgen.evaluate_actions(obs_next, None, None, actions)
                state_log_probs, state_entropy = self.bw_stategen.evaluate_state_actions(obs_next, actions, obs_delta)
                # Calculate Losses.
                entropy_loss = self.args.entropy_coef*(action_entropy+state_entropy)
                if self.args.per_weight:
                    total_loss = -torch.mean(action_log_probs*weights) - torch.mean(state_log_probs*weights) - entropy_loss
                else:
                    total_loss = -torch.mean(action_log_probs) - torch.mean(state_log_probs) - entropy_loss
                self.bw_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bw_params, self.args.max_grad_norm)
                self.bw_optimizer.step()
                avg_loss += total_loss.item()
            avg_loss /= self.args.epoch

            #Now updating the priorities in the PER Buffer. Use Net Value estimates
            if self.args.per_weight:
                with torch.no_grad():
                    value = self.actor_critic.get_value(obs_next_unnormalized, None, None)
                value = torch.clamp(value, min=0)
                self.buffer.update_priorities(idxes, value.squeeze(1).cpu().numpy())

            # Train FW - Model
            if self.args.consistency:
                f_loss = 0
                for _ in range(self.args.epoch):
                    fstate_log_probs, fstate_entropy = self.fw_stategen.evaluate_state_actions(obs, actions, obs_next)
                    fw_loss = -fstate_log_probs.mean() - self.args.entropy_coef*fstate_entropy
                    self.fw_optimizer.zero_grad()
                    fw_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.fw_stategen.parameters(), self.args.max_grad_norm)
                    self.fw_optimizer.step()
                    f_loss += fw_loss.item()
                f_loss /= self.args.epoch
                return avg_loss, f_loss
            else:
                return avg_loss

        elif self.args.consistency:
            return 0.0, 0.0
        else:
            return 0.0

    def train_imitation(self, update):
        """
        Do these steps
        1. Generate Recall traces from bw_model
        2. Do imitation learning using those recall traces
        """
        # maintain list of sampled episodes(batchwise) and append to list. Then do Imitation learning simply
        if self.args.per_weight:
            _ , _ , _ , states, _ , _ = self.sample_batch(self.args.num_states*1000)
        else:
            _ , _ , _ , states = self.sample_batch_noper(self.args.capacity)

        if states is not None:
            states_normalized = np.nan_to_num((states-self.obs_next_mean)/self.obs_next_std)
            with torch.no_grad():
                # self.actor_critic requires un-normalized states
                states = torch.tensor(states, dtype=torch.float32)
                states_normalized = torch.tensor(states_normalized, dtype=torch.float32)
                if self.args.cuda:
                    states = states.cuda()
                    states_normalized = states_normalized.cuda()
                value = self.actor_critic.get_value(states, None, None)
                sorted_indices = value.cpu().numpy().reshape(-1).argsort()[-self.args.num_states:][::-1]
            # Select high value states under currect valuation for target states_next
            states_next = states[sorted_indices.tolist()]
            states_next_normalized = states_normalized[sorted_indices.tolist()]
            mb_actions, mb_states_prev = [], []
            # Sample the Traces
            for step in range(self.args.trace_size):
                with torch.no_grad():
                    # a_mu = self.bw_actgen
                    _, actions, _, _ = self.bw_actgen.act(states_next_normalized, None, None)
                    # Note these actions are normalized as StateGen takes in normalized actions(they were trained this way)
                    delta_state, _ = self.bw_stategen.act(states_next_normalized, actions)
                    # Unnormalize the actions
                    if self.args.cuda:
                        actions = actions*torch.tensor(self.actions_std, dtype=torch.float32).cuda() + torch.tensor(self.actions_mean, dtype=torch.float32).cuda()
                    else:
                        actions = actions*torch.tensor(self.actions_std, dtype=torch.float32) + torch.tensor(self.actions_mean, dtype=torch.float32)
                    # s_t = s_t+1 + Δs_t
                    # delta_state = select_mj(s_mu, s_sigma)
                    if self.args.cuda:
                        delta_state = delta_state*torch.tensor(self.obs_delta_std, dtype=torch.float32).cuda() + torch.tensor(self.obs_delta_mean, dtype=torch.float32).cuda()
                    else:
                        delta_state = delta_state*torch.tensor(self.obs_delta_std, dtype=torch.float32) + torch.tensor(self.obs_delta_mean, dtype=torch.float32)
                    states_prev = states_next + delta_state
                    states_next = states_prev
                    #np.nan_to_num not available in torch
                    states_next_normalized = np.nan_to_num((states_next.cpu().numpy() - self.obs_mean)/self.obs_next_std)
                    states_next_normalized = torch.tensor(states_next_normalized, dtype=torch.float32)
                    if self.args.cuda:
                        states_next_normalized = states_next_normalized.cuda()
                # Add to list
                mb_actions.append(actions.cpu().numpy())
                mb_states_prev.append(states_prev.cpu().numpy())
            # Begin to do Imitation Learning
            mb_actions = torch.tensor(mb_actions, dtype=torch.float32).view(self.args.num_states*self.args.trace_size, -1)
            mb_states_prev = torch.tensor(mb_states_prev, dtype=torch.float32).view(self.args.num_states*self.args.trace_size, -1)
            if self.args.cuda:
                mb_actions = mb_actions.cuda()
                mb_states_prev = mb_states_prev.cuda()
            _, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(mb_states_prev, None, None, mb_actions)
            total_loss = -torch.mean(action_log_probs) - self.args.entropy_coef*dist_entropy
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # Do the Consistency Bit
            if self.args.consistency:
                print('foo')
            else:
                return total_loss.item()
        elif self.args.consistency:
            return 0.0, 0.0
        else:
            return 0.0

    def step(self, obs, actions, rewards, dones, obs_next):
        """
        Add the batch information into the Buffers
        """
        for n in range(self.args.num_processes):
            self.running_episodes[n].append([obs[n], actions[n], rewards[n], obs_next[n]])
        # to see if can update the episode...
        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[n])
                # Clear the episode buffer
                self.running_episodes[n] = []
        if len(self.buffer) >= self.args.capacity -1 :
            # Buffer is full
            if self.buffer._next_idx > int(self.args.ratio*self.args.capacity):
                # Limit reached. Sort and 0
                self.buffer._storage.sort(key=lambda x: x[2])
                self.buffer._next_idx = self.buffer._next_idx % int(self.args.ratio*self.args.capacity)

    def update_buffer(self, trajectory):
        """
        Update buffer. Add single episode to PER Buffer and update stuff
        """
        positive_reward = False
        for (ob, a, r, ob_next) in trajectory:
            if r > 0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            while np.sum(self.total_steps) > self.args.capacity and len(self.total_steps) > 1:
                self.total_steps.pop(0)
                self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        """
        Add single episode to PER Buffer
        """
        obs = []
        actions = []
        rewards = []
        dones = []
        obs_next = []
        for (ob, action, reward, ob_next) in trajectory:
            if ob is not None:
                obs.append(ob)
            else:
                obs.append(None)
            if ob_next is not None:
                obs_next.append(ob_next)
            else:
                obs_next.append(None)
            actions.append(action)
            # rewards.append(np.sign(reward))
            rewards.append(reward)
            dones.append(False)
        # Put done at end of trajectory
        dones[len(dones) - 1] = True
        returns = self.discount_with_dones(rewards, dones, self.args.gamma)
        for (ob, action, R, ob_next) in list(zip(obs, actions, returns, obs_next)):
            self.buffer.add(ob, action, R, ob_next)

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0

    def num_episodes(self):
        return len(self.total_rewards)

    def num_steps(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=self.args.sil_beta)
        else:
            return None, None, None, None, None, None
    
    def sample_batch_noper(self, batch_size):
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size)
        else:
            return None, None, None, None, None, None

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]
