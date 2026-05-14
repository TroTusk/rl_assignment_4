#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from statsmodels.nonparametric.kernel_regression import KernelReg
import os


class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Episode Return')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self, x, y, label=None, y_err=None, **kwargs):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend 
        y_err: vector of standard error for shaded region '''
        
        # plot curve with label if provided
        if label is not None:
            p = self.ax.plot(x, y, label=label, **kwargs)
        else:
            p = self.ax.plot(x, y, **kwargs)
            
        # add shaded error region
        if y_err is not None:
            self.ax.fill_between(x, y - y_err, y + y_err, color=p[0].get_color(), alpha=0.2, edgecolor='none')
    
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def save_csv(eval_timesteps, learning_curve, smoothed_learning_curve, filename):
    os.makedirs("exp_run_results", exist_ok=True)
    df = pd.DataFrame({
    "env_step": eval_timesteps,
    "return": learning_curve,
    "smoothed_return": smoothed_learning_curve
    })
    df.to_csv(f"exp_run_results/{filename}.csv", index=False)


def final_window_score(learning_curve, k=10):
    return learning_curve[-k:].mean()

def smooth(y, window, poly=2):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax 
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)

def linear_anneal(t,T,start,final,percentage):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    ''' 
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T


def load_curve(path):
    df = pd.read_csv(path).sort_values("env_step").reset_index(drop=True)
    col = "smoothed_return" if "smoothed_return" in df.columns else "return"
    return df["env_step"].to_numpy(), df[col].to_numpy()


def compare_plot():
    smoothing_window = 9

    Plot = LearningCurvePlot(title='Algorithm Comparison on CartPole-v1')
    Plot.set_ylim(0, 550)
    Plot.add_hline(height=500, label='Optimal')

    # baseline from previous assignments
    if os.path.exists("BaselineDataCartPole.csv"):
        baseline_df = pd.read_csv("BaselineDataCartPole.csv").sort_values("env_step").reset_index(drop=True)
        baseline_timesteps = baseline_df["env_step"].to_numpy()
        baseline_curve = baseline_df["Episode_Return"].to_numpy()

        target_steps = np.arange(20000, 960001, 10000)
        mask = np.isin(baseline_timesteps, target_steps)
        baseline_eval_timesteps = baseline_timesteps[mask]
        baseline_learning_curve = baseline_curve[mask]
        baseline_learning_curve = smooth(baseline_learning_curve, smoothing_window)
        baseline_learning_curve = np.clip(baseline_learning_curve, 0, 500)

        Plot.add_curve(baseline_eval_timesteps, baseline_learning_curve, label='Baseline', color='black', linestyle='--', linewidth=1.1)

    # previous algorithms to compare results
    files = [
        ("compare_results/Naive_DQN_best_param.csv", "Naive DQN"),
        ("compare_results/REINFORCE.csv", "REINFORCE"),
        ("compare_results/AC.csv", "AC"),
        ("compare_results/A2C.csv", "A2C"),
    ]
    for path, label in files:
        if os.path.exists(path):
            x, y = load_curve(path)
            Plot.add_curve(x, y, label=label)

    # PPO from current run
    if os.path.exists("exp_run_results/PPO.csv"):
        x, y = load_curve("exp_run_results/PPO.csv")
        Plot.add_curve(x, y, label='PPO')

    Plot.save('PPO_comparison.png')


if __name__ == '__main__':
    # Test Learning curve plot
    x = np.arange(100)
    y = 0.01*x + np.random.rand(100) - 0.4 # generate some learning curve y
    LCTest = LearningCurvePlot(title="Test Learning Curve")
    LCTest.add_curve(y,label='method 1')
    LCTest.add_curve(smooth(y,window=35),label='method 1 smoothed')
    LCTest.save(name='learning_curve_test.png')