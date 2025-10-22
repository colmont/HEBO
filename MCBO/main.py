import os
import time
import gin
import argparse
import logging

from experiments.run_task_exps import Experiments
from mcbo.utils.experiment_utils import gin_config_to_omegaconf

CONFIG_PATH = "configs/default.gin"
SEED = 42

if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "True"

    then = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gin-files",
        type=str,
        nargs="+",
        default=[CONFIG_PATH],
        help="Path to the config file",
    )
    parser.add_argument(
        "--gin-bindings",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_files, args.gin_bindings)


    wandb_config = gin_config_to_omegaconf(CONFIG_PATH)
    wandb_config["seed"] = args.seed

    Experiments(wandb_config=wandb_config, seed=args.seed).run()

    gin.clear_config()
    now = time.time()
    logging.info(f"Total time: {now - then:.2f} seconds")
