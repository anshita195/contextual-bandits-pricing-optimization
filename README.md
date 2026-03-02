# Contextual Multi-Armed Bandits for Online Markdown Pricing Optimization

## Abstract

This project implements a reinforcement learning framework designed to optimize markdown pricing strategies in e-commerce environments. By utilizing Contextual Multi-Armed Bandit (CMAB) algorithms—specifically the Linear Upper Confidence Bound (LinUCB) agent—the system learns to dynamically adjust product prices based on real-time customer and product contexts. The project demonstrates the transition from theoretical bandit algorithms to real-world deployment through a two-phase analysis involving the UCI Online Retail and Olist Brazilian E-commerce datasets.

## Project Overview

The "Exploration vs. Exploitation" dilemma is central to dynamic pricing. A system must exploit known profitable price points while exploring new discounts to adapt to changing demand. This project replaces static A/B testing with an automated agent that minimizes regret—the difference between the optimal profit and the profit obtained by the algorithm.

## Core Algorithms

### 1. Linear Upper Confidence Bound (LinUCB)

The primary production algorithm, which assumes the expected reward is a linear function of the context. It utilizes ridge regression to estimate coefficients ($\theta$) and calculates an uncertainty bonus to drive exploration.

### 2. Multi-Armed Bandit Variants

The repository includes several standard MAB implementations for comparative analysis:

* **Epsilon-Greedy**: A simple balance using a configurable epsilon parameter to decide between random exploration and greedy exploitation.
* **Thompson Sampling**: A Bayesian approach using probability distributions (Bernoulli, Gaussian, and Gamma) to model reward uncertainty.
* **Upper Confidence Bound (UCB)**: Includes UCB1, UCB2 for better theoretical guarantees, and UCBV for variance estimation.
* **Softmax (Boltzmann Exploration)**: Arm selection based on a temperature-scaled probability distribution.
* **Gradient Bandit**: A policy gradient approach that learns arm selection probabilities directly.

## Feature Engineering (Contextual Vectors)

To enable the bandit to learn effectively, raw data is transformed into high-dimensional vectors:

* **RFM Metrics**: Recency, Frequency, and Monetary scores are used to segment customer value and behavior.
* **Temporal Features**: Normalized indicators for weeks, months, and days to capture seasonal demand patterns.
* **Logistics & Metadata**: Integration of freight-to-price ratios and delivery performance metrics specific to the Olist dataset.
* **Payment & Sentiment**: Contextual data regarding payment types, installment buckets, and customer review scores.

## Project Structure

```text
multi_armed_bandit/
├── algorithms/           # Core MAB and LinUCB implementations
├── applications/         # Simulators for Advertising, Clinical Trials, and Recommenders
├── utils/                # Environment simulations and performance metrics
├── phase1_final_analysis.py # UCI dataset validation and statistical testing
├── phase2_olist_working.py  # Large-scale Olist integration and reward scaling
├── uci_retail_pipeline.py   # Class-based production-ready pipeline
├── main.py               # Central CLI for running experiments
└── requirements.txt      # Project dependencies

```

## Experimental Results

### Phase 1: UCI Retail Validation

* **Profit Improvement**: The LinUCB agent achieved a 19.4% improvement in simulated profit over the baseline pricing strategy.
* **Statistical Significance**: Improvements were verified through weekly aggregated profits using paired t-tests and Cohen's d metrics.

### Phase 2: Olist Integration

* **Contextual Gain**: Integrating payment and shipping data as context led to a 21.7% profit improvement over baseline models.
* **Reward Scaling**: Implementation of log-scaled reward functions allowed the model to handle high-variance price points effectively.

### Real-World Simulations

* **Online Advertising**: The Epsilon-Greedy variant ($\epsilon=0.2$) achieved 85.3% efficiency in ad selection optimization.
* **Clinical Trials**: Demonstration of patient allocation optimization using UCB to balance ethical considerations with drug efficacy testing.

## Installation & Usage

1. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


2. **Run Pipeline Analysis**:
```bash
python phase1_final_analysis.py
python phase2_olist_working.py

```


3. **Execute CLI Experiments**:
```bash
# Basic algorithm comparison
python main.py --mode basic --arms 10 --trials 1000

# Real-world simulation mode
python main.py --mode real-world

```
