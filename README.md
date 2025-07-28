# Multi-Armed Bandit Project

A comprehensive implementation of Multi-Armed Bandit (MAB) algorithms with real-world applications and interactive visualizations.

## 🎯 Project Overview

This project implements various Multi-Armed Bandit algorithms to solve the exploration vs exploitation dilemma. The MAB problem is fundamental in reinforcement learning and has applications in online advertising, clinical trials, recommender systems, and more.

## 🚀 Features

### Core Algorithms
- **Epsilon-Greedy**: Simple exploration-exploitation balance
- **Upper Confidence Bound (UCB)**: Optimism in the face of uncertainty
- **Thompson Sampling**: Bayesian approach with probability distributions
- **Softmax**: Temperature-based arm selection
- **Gradient Bandit**: Policy gradient approach

### Real-World Applications
- **Online Advertising Simulator**: A/B testing for ad campaigns
- **Clinical Trial Simulator**: Drug efficacy testing
- **Recommender System**: Movie recommendation engine
- **Network Routing**: Adaptive path selection

### Visualization & Analysis
- Interactive performance comparisons
- Regret analysis
- Arm selection frequency tracking
- Real-time reward visualization

## 📁 Project Structure

```
multi_armed_bandit/
├── algorithms/
│   ├── __init__.py
│   ├── epsilon_greedy.py
│   ├── ucb.py
│   ├── thompson_sampling.py
│   ├── softmax.py
│   └── gradient_bandit.py
├── applications/
│   ├── __init__.py
│   ├── advertising_simulator.py
│   ├── clinical_trial.py
│   ├── recommender_system.py
│   └── network_routing.py
├── visualization/
│   ├── __init__.py
│   ├── performance_plots.py
│   └── interactive_dashboard.py
├── utils/
│   ├── __init__.py
│   ├── environment.py
│   └── metrics.py
├── examples/
│   ├── basic_comparison.py
│   ├── advertising_demo.py
│   └── clinical_trial_demo.py
├── requirements.txt
├── main.py
└── README.md
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multi_armed_bandit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Basic Usage
```python
from algorithms import EpsilonGreedy, UCB, ThompsonSampling
from utils.environment import BanditEnvironment

# Create environment
env = BanditEnvironment(n_arms=10, n_trials=1000)

# Test different algorithms
algorithms = {
    'Epsilon-Greedy': EpsilonGreedy(n_arms=10, epsilon=0.1),
    'UCB': UCB(n_arms=10),
    'Thompson Sampling': ThompsonSampling(n_arms=10)
}

# Run experiments
results = env.run_experiments(algorithms)
```

### Real-World Applications
```python
# Online Advertising
from applications.advertising_simulator import AdvertisingSimulator
simulator = AdvertisingSimulator()
simulator.run_campaign()

# Clinical Trials
from applications.clinical_trial import ClinicalTrial
trial = ClinicalTrial()
trial.run_trial()
```

## 📊 Results & Analysis

The project includes comprehensive analysis tools:
- Performance comparison charts
- Regret analysis
- Statistical significance testing
- Interactive dashboards

## 🎯 Key Learning Objectives

1. **Understanding Exploration vs Exploitation**: Learn how to balance trying new options vs exploiting known good options
2. **Algorithm Comparison**: Compare different MAB strategies
3. **Real-World Applications**: Apply MAB concepts to practical problems
4. **Performance Analysis**: Learn to evaluate and compare algorithms

## 📚 Educational Value

This project is perfect for:
- Students learning reinforcement learning
- Data scientists exploring MAB algorithms
- Engineers implementing recommendation systems
- Researchers studying decision-making under uncertainty

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 