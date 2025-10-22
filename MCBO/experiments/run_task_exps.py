import os
import sys
import signal
from pathlib import Path

import torch
import zlib
import gin
import wandb
from omegaconf import OmegaConf

from mcbo.utils.experiment_utils import (
    run_experiment,
    get_task_from_id,
    get_opt,
)

sys.path.insert(0, str(Path(os.path.realpath(__file__)).parent.parent))

NUM_CAT_FOR_SEARCH_SPACE = {
    "ackley": 11,
    "schwefel": 11,
    "rastrigin": 11,
    "sphere": 11,
    "styblinski_tang": 11,
    "pest": 5,
    "rna_inverse_fold": 4,
    "antibody_design": 20,
    "aig_optimization": 10,
    "mig_optimization": 7,
}

DTYPE = torch.float64
ABSOLUT_DIR = "/home/HEBO/MCBO/libs/Absolut/src/AbsolutNoLib"


@gin.configurable
class Experiments:

    def __init__(
        self,
        wandb_config: dict,
        seed: int,
        device_id: int = 0,
        absolut_dir: str = ABSOLUT_DIR,
        result_dir: str = "./results",
        verbose: int = 2,
        debug: bool = True,
        task_id: str = "ackley",
        opt_id: str = "gp_oh_rbf__ga__ei__basic",
        bo_n_init: int = 20,
        max_num_iter: int = 200,
    ):
        self.wandb_config = wandb_config
        self.device_id = device_id
        self.absolut_dir = absolut_dir
        self.verbose = verbose
        self.debug = debug
        self.seed = seed
        self.task_id = task_id
        self.opt_id = opt_id
        self.bo_n_init = bo_n_init
        self.max_num_iter = max_num_iter

        # defining results directory
        gin_config_str = gin.config_str()
        adler = zlib.adler32(gin_config_str.encode("utf-8"))
        self.result_dir = os.path.join(result_dir, str(adler))
        os.makedirs(self.result_dir, exist_ok=True)

        # save gin config to file
        with open(os.path.join(self.result_dir, "gin_config.txt"), "w") as f:
            f.write(gin.config_str())

    def _init_wandb(self, task_id, debug, wandb_config):
        project_name = f"MCBO-{task_id}"
        wandb_mode = "disabled" if debug else "online"
        tracker = wandb.init(
            project=project_name,
            entity="",
            config=OmegaConf.to_container(wandb_config, resolve=True),
            mode=wandb_mode,
        )
        return self._handle_interrupt(tracker)

    def _handle_interrupt(self, tracker):
        """Handles keyboard interrupt to ensure graceful tracker termination."""

        def signal_handler(signum, frame):
            print("Ctrl-C pressed, terminating tracker...")
            tracker.finish()
            print("Tracker terminated. Exiting.")
            sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)
        return tracker

    def run(self):
        if self.device_id >= 0 and torch.cuda.is_available():
            bo_device_ = torch.device(f"cuda:{self.device_id}")
        else:
            bo_device_ = torch.device("cpu")

        self.tracker = self._init_wandb(self.task_id, self.debug, self.wandb_config)

        task = get_task_from_id(task_id=self.task_id, absolut_dir=self.absolut_dir)
        opt = get_opt(
            task=task,
            short_opt_id=self.opt_id,
            bo_n_init=self.bo_n_init,
            dtype=DTYPE,
            bo_device=bo_device_,
        )
        run_experiment(
            task=task,
            optimizers=[opt],
            random_seeds=[self.seed],
            max_num_iter=self.max_num_iter,
            save_results_every=self.max_num_iter,
            very_verbose=self.verbose > 1,
            tracker=self.tracker,
            result_dir=self.result_dir,
        )

        self.tracker.finish()
