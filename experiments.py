import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from Agent import train_ppo
from Helper import LearningCurvePlot, smooth, save_csv
import numpy as np



def average_over_repetitions(n_repetitions, smoothing_window, n_timesteps, learning_rate, gamma, eval_interval, net_size,
                             clip_eps, K_epochs, batch_size, gae_lambda, rollout_steps):

    returns_over_repetitions = []
    eval_timesteps = None

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(train_ppo, n_timesteps, learning_rate, gamma, eval_interval, net_size,
                            clip_eps, K_epochs, batch_size, gae_lambda, rollout_steps)
            for _ in range(n_repetitions)
        ]

        for future in as_completed(futures):
            eval_returns, rep_eval_timesteps = future.result()
            returns_over_repetitions.append(eval_returns)
            if eval_timesteps is None:
                eval_timesteps = rep_eval_timesteps

    # calculate mean and standard error across repetitions
    returns_matrix = np.array(returns_over_repetitions)
    learning_curve = np.mean(returns_matrix, axis=0)
    standard_error = np.std(returns_matrix, axis=0) / np.sqrt(n_repetitions)

    if smoothing_window is not None:
        smoothed_learning_curve = smooth(learning_curve, smoothing_window)
        smoothed_se = smooth(standard_error, smoothing_window)
        return learning_curve, smoothed_learning_curve, smoothed_se, eval_timesteps

    return learning_curve, learning_curve, standard_error, eval_timesteps


def run_experiments():
    n_repetitions = 5
    n_timesteps = 1000001 # 100000
    eval_interval = 10000
    smoothing_window = 9

    gamma = 0.99
    learning_rate = 0.001
    net_size = 64

    clip_eps = 0.2
    K_epochs = 4
    batch_size = 64
    gae_lambda = 0.95
    rollout_steps = 2048

    Plot = LearningCurvePlot(title='PPO Learning Curve on CartPole-v1')
    Plot.set_ylim(0, 550)
    Plot.add_hline(height=500, label='Optimal')

    
    print("PPO Experiment")

    ppo_curve, ppo_smoothed, ppo_se, ppo_steps = average_over_repetitions(
        n_repetitions, smoothing_window, n_timesteps, learning_rate, gamma, eval_interval, net_size,
        clip_eps, K_epochs, batch_size, gae_lambda, rollout_steps)

    save_csv(ppo_steps, ppo_curve, ppo_smoothed, filename='PPO')
    Plot.add_curve(ppo_steps, ppo_smoothed, label='PPO', y_err=ppo_se)

    Plot.save('PPO_learning_curve.png')


if __name__ == '__main__':
    run_experiments()