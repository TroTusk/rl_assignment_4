import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm


#PPO
class PPOActorNetwork(nn.Module):
    """Actor network: outputs a probability distribution of the actions"""
    def __init__(self, n_observations, n_actions, net_size):
        super(PPOActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observations, net_size),
            nn.ReLU(),
            nn.Linear(net_size, net_size),
            nn.ReLU(),
            nn.Linear(net_size, n_actions)
        )

    def forward(self, x):
        return self.network(x)


class PPOCriticNetwork(nn.Module):
    """Critic network: outputs a scalar state value V(s)"""
    def __init__(self, n_observations, net_size):
        super(PPOCriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_observations, net_size),
            nn.ReLU(),
            nn.Linear(net_size, net_size),
            nn.ReLU(),
            nn.Linear(net_size, 1)
        )

    def forward(self, x):
        return self.network(x)


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    """

    def __init__(self, n_observations, n_actions, learning_rate, gamma,
                 clip_eps, K_epochs, batch_size,
                 gae_lambda, net_size=64):

        self.gamma = gamma
        self.clip_eps = clip_eps # clipping range for the probability ratio
        self.K_epochs = K_epochs # number of update epochs per rollout
        self.batch_size = batch_size # minibatch size
        self.gae_lambda = gae_lambda # lambda for GAE

        self.actor = PPOActorNetwork(n_observations, n_actions, net_size)
        self.critic = PPOCriticNetwork(n_observations, net_size)

        # optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate)

    def select_action(self, state, evaluate=False):
        # convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # compute action logits from actor
        with torch.no_grad():
            logits = self.actor(state_tensor)

        # greedy action for evaluation
        if evaluate:
            action = torch.argmax(logits)
            return int(action.item()), None, None
        
        # sample action from policy during training
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # also return state value needed for GAE computation at collection time
        with torch.no_grad():
            value = self.critic(state_tensor).squeeze()
        return int(action.item()), log_prob.item(), value.item()

    # Do evaluations every interval
    def evaluate(self, eval_env, n_eval_episodes=10):
        returns = []
        for _ in range(n_eval_episodes):
            state, _ = eval_env.reset()
            episode_return = 0
            done = False
            while not done:
                action = self.select_action(state, evaluate=True)[0]
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_return += reward
                state = next_state
            returns.append(episode_return)


        return np.mean(returns)

    def _compute_gae(self, rewards, values, dones, last_value):
        """
        Generalized Advantage Estimation
        delta_t = r_t + gamma * V(s_t+1) * (1 - done) - V(s_t)
        A_t = sum_(k=0)^(T-t) (gamma * lambda)^k * delta_t+k
        """
        advantages = []
        gae = 0.0
        values_extended = values + [last_value]

        # iterate backwards since GAE is computed recursively from the last step
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values_extended[t + 1] * mask - values_extended[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae) # insert at front to restore chronological order
        # returns = advantages + values (used as critic targets)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(self, states, actions, old_log_probs, rewards, dones, last_value):
        """
        PPO-clip update over K epochs with minibatches
        old_log_probs is a tuple (values_list, log_probs_list)
        """
        values_list, old_log_probs_list = old_log_probs
        advantages, returns = self._compute_gae(rewards, values_list, dones, last_value)

        # convert everything to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        old_lp_t = torch.tensor(old_log_probs_list, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        returns_t= torch.tensor(returns, dtype=torch.float32)

        # normalize advantages to reduce variance
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = len(states)
        for _ in range(self.K_epochs):
            # shuffle indices for minibatch sampling
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start: start + self.batch_size]
                sb = states_t[idx]
                ab = actions_t[idx]
                old_lpb = old_lp_t[idx]
                adv_b = advantages_t[idx]
                ret_b = returns_t[idx]

                # recompute log-probs with current policy
                logits = self.actor(sb)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(ab)

                # probability ratio r_t(theta) = pi_theta / pi_theta_old
                ratio = torch.exp(new_log_probs - old_lpb)

                # clipped surrogate objective
                # surr1: unclipped objective r_t * A_t (standard policy gradient term)
                surr1 = ratio * adv_b
                # surr2: same objective but with ratio clipped to [1-eps, 1+eps]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                # take the minimum to get a pessimistic bound, then negate for gradient descent
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic MSE loss
                values_pred = self.critic(sb).squeeze()
                critic_loss = nn.MSELoss()(values_pred, ret_b)

                # combined loss
                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                # gradient clipping to avoid excessively large updates
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=0.5)
                
                # apply the clipped gradients to update actor and critic weights
                self.optimizer.step()


# training loop 
def train_ppo(n_timesteps, learning_rate, gamma, eval_interval=1000, net_size=64,
              clip_eps=0.2, K_epochs=4, batch_size=64, gae_lambda=0.95,
              rollout_steps=2048):
    """
    PPO training loop.

    Collects rollout_steps of experience, then updates the policy
    updates do not depend on episode length
    """
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(n_observations, n_actions, learning_rate, gamma,
                     clip_eps, K_epochs, batch_size, gae_lambda, net_size)

    eval_timesteps, eval_returns = [], []
    total_t = 0
    pbar = tqdm(total=n_timesteps, desc="PPO", leave=False)

    # eval at start
    mean_return = agent.evaluate(eval_env)
    eval_returns.append(mean_return)
    eval_timesteps.append(0)

    state, _ = env.reset()

    while total_t < n_timesteps:
        # rollout collection 
        states, actions, log_probs_list, values_list = [], [], [], []
        rewards, dones = [], []

        for _ in range(rollout_steps):
            # sample action, log_prob and state value from the current policy
            action, log_prob, value = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store transition data for the upcoming PPO update
            states.append(state)
            actions.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)
            rewards.append(reward)
            dones.append(float(done))

            total_t += 1

            # periodic evaluation on a separate environment
            if total_t % eval_interval == 0:
                mean_return = agent.evaluate(eval_env)
                eval_returns.append(mean_return)
                eval_timesteps.append(total_t)

            # reset on episode end but keep filling the same rollout buffer
            if done:
                state, _ = env.reset()
            else:
                state = next_state

            if total_t >= n_timesteps:
                break

        pbar.update(len(rewards))
        # bootstrap value for the last state (needed for GAE)
        with torch.no_grad():
            last_value = agent.critic(torch.FloatTensor(state).unsqueeze(0)).item()

        # update
        agent.update(states, actions, (values_list, log_probs_list), rewards, dones, last_value)

    pbar.close()
    env.close()
    eval_env.close()

    return np.array(eval_returns), np.array(eval_timesteps)