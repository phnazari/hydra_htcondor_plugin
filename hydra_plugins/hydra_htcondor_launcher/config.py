# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class BaseQueueConf:
    """Configuration shared by all executors"""

    # the executable
    executable: str = "test.py"

    # the arguments passed to the executable as a string
    arguments: Optional[str] = None

    #htcondor_folder: str = "${hydra.sweep.dir}/.htcondor/%j"

    # the error directory
    error: str = "${hydra.sweep.dir}/.htcondor/%j/err"
    # the output directory
    output: str = "${hydra.sweep.dir}/.htcondor/%j/out"
    # the log directory
    log: str = "${hydra.sweep.dir}/.htcondor/%j/log"

    request_memory: str = "4000"

    request_cpus: str = "1"

    request_gpus: str = "0"

    requirements: Optional[str] = None

    # maximum time for the job in minutes
    MaxTime: int = 8 * 60


@dataclass
class HTCondorQueueConf(BaseQueueConf):
    """HTCondor configuration overrides and specific parameters"""

    _target_: str = (
        "hydra_plugins.hydra_htcondor_launcher.htcondor_launcher.HTCondorLauncher"
    )


# @dataclass
# class LocalQueueConf(BaseQueueConf):
#    _target_: str = (
#        "hydra_plugins.hydra_htcondor_launcher.htcondor_launcher.LocalLauncher"
#    )


# finally, register two different choices:
# ConfigStore.instance().store(
#    group="hydra/launcher",
#    name="submitit_local",
#    node=LocalQueueConf(),
#    provider="submitit_launcher",
# )


ConfigStore.instance().store(
    group="hydra/launcher",
    name="htcondor_launcher",
    node=HTCondorQueueConf(),
    provider="htcondor_launcher",
)
