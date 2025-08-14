# AIXueRLHF

A variant of PPO to learn from $\langle s, a, r \rangle$ tuples.

## Description

This project implements a variant of Proximal Policy Optimization (PPO) algorithm for reinforcement learning from human feedback (RLHF).

## Features

- PPO variant implementation
- Learning from state-action-reward tuples
- Human feedback integration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd AIXueRLHF
sh ./scripts/train_example.sh
```

The example dataset is ./data/aixue_test_data with 480 $\langle \text{prompt}, \text{response}, \text{reward} \rangle$ pairs.

You can find the trained model and the logged tensorboard in "your_output_dir".

Don't forget to change the "num_processes" in ./configs/deepspeed_zero3.yaml.

## License

[Add your license here]
