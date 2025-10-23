# Using the `MCBO` benchmark

## Introduction
This is a fork from the original `MCBO` repository, with minimal changes to incorporate the proposed heat kernel, as well as run experiments more smoothly. Our modificiations to the original repo should be visible in the commit history.

## Installation
To install the required dependencies, please refer to the [original README.md file](./README_original.md). Our fork only needs a few additional dependencies that can be installed as follows: `pip install omegaconf gin-config wandb`.

## Usage
To run the BO loop, execute the following command:
```bash
python main.py 
```
which will use the values from the associated [config file](./configs/default.gin). All experiments will be saved automatically using Weights & Biases, and large parallel experiments can be run using the associated [bash script](./cpu_parallel.sh), which contains all the experiments from the paper.