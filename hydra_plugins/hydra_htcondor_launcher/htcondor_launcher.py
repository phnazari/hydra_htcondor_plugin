# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    JobStatus,
    filter_overrides,
    run_job,
    setup_globals,
)
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict

from .config import BaseQueueConf

log = logging.getLogger(__name__)


class HTCondorLauncher(Launcher):
    """HTCondor launcher for Hydra multirun jobs using HTCondor Python bindings."""

    def __init__(self, **params: Any) -> None:
        self.params = {}
        for k, v in params.items():
            if OmegaConf.is_config(v):
                v = OmegaConf.to_container(v, resolve=True)
            self.params[k] = v

        # Debug: print the loaded parameters
        log.info(f"HTCondor launcher initialized with params: {self.params}")

        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        """Launch jobs using HTCondor."""
        # lazy import to ensure plugin discovery remains fast
        import htcondor2 as htcondor

        assert self.config is not None
        assert self.hydra_context is not None
        assert self.task_function is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0

        log.info(f"HTCondor launcher submitting {num_jobs} jobs")
        log.info(f"Sweep output dir: {self.config.hydra.sweep.dir}")

        # Create sweep directory
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        log.info("Submitting jobs to HTCondor")
        log.info(
            f"HTCondor config: memory={self.params.get('request_memory', '4000')}MB, "
            f"cpus={self.params.get('request_cpus', '1')}, "
            f"gpus={self.params.get('request_gpus', '0')}"
        )

        # Build HTCondor executor
        htcondor_folder = self.params.get(
            "htcondor_folder", "${hydra.sweep.dir}/.htcondor"
        )
        htcondor_folder = htcondor_folder.replace("${hydra.sweep.dir}", str(sweep_dir))
        htcondor_dir = Path(htcondor_folder)
        htcondor_dir.mkdir(parents=True, exist_ok=True)

        # Create job parameters like submitit does
        job_params: List[Any] = []
        for idx, overrides in enumerate(job_overrides):
            job_idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{job_idx} : {lst}")
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    job_idx,
                    f"job_id_for_{job_idx}",
                    Singleton.get_state(),
                )
            )

        # Create HTCondor executor
        executor = self._create_htcondor_executor(htcondor, sweep_dir)

        # Submit jobs using map_array pattern like submitit
        jobs = executor.map_array(job_params)

        # Return results like submitit does
        return [j.result() for j in jobs]

    def __call__(
        self,
        sweep_overrides: List[str],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: Dict[type, Singleton],
    ) -> JobReturn:
        """Execute a single job - called by HTCondor on compute nodes."""
        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None

        Singleton.set_state(singleton_state)
        setup_globals()

        sweep_config = self.hydra_context.config_loader.load_sweep_config(
            self.config, sweep_overrides
        )

        with open_dict(sweep_config.hydra.job) as job:
            job.id = job_id
            job.num = job_num

        return run_job(
            hydra_context=self.hydra_context,
            task_function=self.task_function,
            config=sweep_config,
            job_dir_key=job_dir_key,
            job_subdir_key="hydra.sweep.subdir",
        )

    def _create_htcondor_executor(self, htcondor, sweep_dir):
        """Create HTCondor executor following submitit pattern."""
        # Create HTCondor working directory using config or default
        htcondor_folder = self.params.get(
            "htcondor_folder", "${hydra.sweep.dir}/.htcondor"
        )
        # Resolve the hydra variable
        htcondor_folder = htcondor_folder.replace("${hydra.sweep.dir}", str(sweep_dir))
        htcondor_dir = Path(htcondor_folder)
        htcondor_dir.mkdir(exist_ok=True)

        return HTCondorExecutor(htcondor_dir, self.params, htcondor)


class HTCondorJob:
    """HTCondor job wrapper for tracking and result collection."""

    def __init__(
        self, cluster_id, job_id, htcondor_module, log_file, output_file, error_file
    ):
        self.cluster_id = cluster_id
        self.job_id = job_id
        self.htcondor = htcondor_module
        self.log_file = Path(log_file)
        self.output_file = Path(output_file)
        self.error_file = Path(error_file)

    def result(self, timeout=None):
        """Wait for job completion and return JobReturn."""
        import time

        start_time = time.time()

        # Poll job status until completion
        schedd = self.htcondor.Schedd()

        while True:
            # Check if timeout exceeded
            if timeout and (time.time() - start_time) > timeout:
                result = JobReturn()
                result.status = JobStatus.FAILED
                result.exception = TimeoutError(
                    f"Job {self.cluster_id}.{self.job_id} timed out after {timeout}s"
                )
                return result

            # Query job status
            try:
                jobs = list(
                    schedd.query(
                        f"ClusterId == {self.cluster_id} && ProcId == {self.job_id}"
                    )
                )
                if not jobs:
                    # Job not found, might be completed and cleaned up
                    break

                job = jobs[0]
                job_status = job.get("JobStatus", 0)

                # HTCondor job status codes:
                # 1 = Idle, 2 = Running, 3 = Removed, 4 = Completed, 5 = Held, 6 = Transferring output
                if job_status in [4, 3]:  # Completed or Removed
                    break
                elif job_status == 5:  # Held
                    result = JobReturn()
                    result.status = JobStatus.FAILED
                    hold_reason = job.get("HoldReason", "Unknown hold reason")
                    result.exception = RuntimeError(f"Job held: {hold_reason}")
                    return result

            except Exception as e:
                log.warning(f"Error querying job status: {e}")

            time.sleep(5)  # Poll every 5 seconds

        # Job completed, check exit code and create result
        result = JobReturn()

        # Read exit code from log file if available
        exit_code = 0
        if self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    content = f.read()
                    # Look for exit code in HTCondor log
                    import re

                    match = re.search(
                        r"Job terminated\.\s+\(.*\)\s+Normal termination \(return value (\d+)\)",
                        content,
                    )
                    if match:
                        exit_code = int(match.group(1))
            except Exception as e:
                log.warning(f"Could not read exit code from log file: {e}")

        if exit_code == 0:
            result.status = JobStatus.COMPLETED
        else:
            result.status = JobStatus.FAILED
            # Read error output if available
            error_msg = f"Job failed with exit code {exit_code}"
            if self.error_file.exists():
                try:
                    with open(self.error_file, "r") as f:
                        stderr_content = f.read().strip()
                        if stderr_content:
                            error_msg += f"\nStderr: {stderr_content}"
                except Exception:
                    pass
            result.exception = RuntimeError(error_msg)

        return result


class HTCondorExecutor:
    """HTCondor executor following submitit pattern."""

    def __init__(self, folder, params, htcondor_module):
        self.folder = Path(folder)
        self.params = params
        self.htcondor = htcondor_module

    def map_array(self, job_params):
        """Submit array of jobs to HTCondor."""
        jobs = []
        schedd = self.htcondor.Schedd()

        # Get output directory from config
        output_dir = self.params.get("output_dir", str(self.folder))

        for job_param in job_params:
            overrides, _, job_idx, job_id, _ = job_param
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{job_idx} : {lst}")

            # Create override string for command line
            override_str = " ".join([f"'{override}'" for override in overrides])

            # Create command that runs the same script with specific overrides
            cmd_args = f"-m example.my_app {override_str}"

            # Create unique file names for this job
            job_output = f"{output_dir}/job_{job_idx}.out"
            job_error = f"{output_dir}/job_{job_idx}.err"
            job_log = f"{output_dir}/job_{job_idx}.log"

            # Create HTCondor submit description using config values
            submit_dict = {
                "executable": self.params.get("executable", sys.executable),
                "arguments": cmd_args,
                "output": job_output,
                "error": job_error,
                "log": job_log,
                "request_memory": str(self.params.get("request_memory", "4000")),
                "request_cpus": str(self.params.get("request_cpus", "1")),
                "request_gpus": str(self.params.get("request_gpus", "0")),
                "should_transfer_files": "YES",
                "when_to_transfer_output": "ON_EXIT",
                "getenv": "True",
            }

            # Add requirements if specified in config
            if "requirements" in self.params:
                submit_dict["requirements"] = str(self.params["requirements"])

            # Add MaxTime and periodic_remove if specified
            if "MaxTime" in self.params:
                submit_dict["MaxTime"] = str(self.params["MaxTime"])
                # Add periodic remove based on MaxTime
                submit_dict["periodic_remove"] = (
                    f"(JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= {self.params['MaxTime']})"
                )

            # Add any additional custom parameters from config
            reserved_keys = {
                "executable",
                "arguments",
                "output",
                "error",
                "log",
                "request_memory",
                "request_cpus",
                "request_gpus",
                "should_transfer_files",
                "when_to_transfer_output",
                "getenv",
                "use_htcondor",
                "output_dir",
                "htcondor_folder",
                "requirements",
                "MaxTime",
            }
            for key, value in self.params.items():
                if key not in reserved_keys:
                    submit_dict[key] = str(value)

            # Submit the job
            submit_obj = self.htcondor.Submit(submit_dict)
            submit_result = schedd.submit(submit_obj)

            cluster_id = submit_result.cluster()
            log.info(f"Submitted job {job_idx} as HTCondor job {cluster_id}.0")

            # Create HTCondorJob wrapper for tracking
            htcondor_job = HTCondorJob(
                cluster_id=cluster_id,
                job_id=0,  # Single job per cluster in this case
                htcondor_module=self.htcondor,
                log_file=job_log,
                output_file=job_output,
                error_file=job_error,
            )
            jobs.append(htcondor_job)

        return jobs
