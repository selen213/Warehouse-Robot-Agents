# Warehouse Robot Agents

A turn-based warehouse simulation in which two robots compete to collect and deliver packages while managing battery and credit.

The project includes a game environment, graphical visualization, baseline agents, and interfaces for implementing adversarial search.

## Included Components

- Warehouse environment with robots, packages, and charging stations.
- Random and greedy baseline agents.
- Pygame visualization.
- Configurable per-turn time limits.
- Reproducible games using random seeds.
- A tournament mode for comparing agents.

## Implementation Status

The current version includes unfinished templates for:

- An improved evaluation heuristic.
- Minimax.
- Alpha-beta pruning.
- Expectimax.

These methods are not implemented in the supplied `submission.py`.

## Files

| File | Purpose |
|---|---|
| `WarehouseEnv.py` | Game state, rules, and visualization |
| `Agent.py` | Base agent and random/greedy baselines |
| `submission.py` | Custom agent implementation templates |
| `main.py` | Command-line game runner |
| `icons/` | Visualization assets |

## Installation

    pip install pygame

## Run a Baseline Match

    python main.py random greedy -s 42 -c 100 --console_print

## Run with Visualization

    python main.py random greedy -s 42 -c 100 --screen_print

## Academic Context

This repository contains an AI assignment environment and agent templates. Course-provided framework code should be distinguished from student-written agent implementations.
