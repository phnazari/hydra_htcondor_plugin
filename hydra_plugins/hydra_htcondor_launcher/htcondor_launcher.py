# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides, run_job, setup_globals
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
        import htcondor

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

        # Create HTCondor working directory
        htcondor_dir = sweep_dir / ".htcondor"
        htcondor_dir.mkdir(exist_ok=True)

        # Build executor similar to submitit pattern
        executor = self._build_executor(htcondor_dir)

        # Submit jobs
        job_params = []
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

        # Submit all jobs using the executor
        jobs = executor.map_array(self, *zip(*job_params))

        # Collect results
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

    def _build_executor(self, htcondor_dir: Path):
        """Build HTCondor executor similar to submitit pattern."""
        # lazy import to ensure plugin discovery remains fast
        import htcondor

        # Create a simple HTCondor executor
        class HTCondorExecutor:
            def __init__(self, htcondor_dir: Path, params: Dict[str, Any]):
                self.htcondor_dir = htcondor_dir
                self.params = params
                self.schedd = htcondor.Schedd()

            def map_array(self, func, *iterables):
                """Submit jobs as an HTCondor job array."""
                # Create a single submit description for all jobs
                submit_desc = self._create_submit_description(
                    len(list(zip(*iterables)))
                )

                # Store job parameters for each process
                for i, args in enumerate(zip(*iterables)):
                    self._store_job_params(i, func, args)

                # Submit the job array
                with self.schedd.transaction() as txn:
                    submit_result = submit_desc.queue(
                        txn, count=len(list(zip(*iterables)))
                    )
                    cluster_id = submit_result.cluster()

                # Create job objects for result collection
                jobs = []
                for i in range(len(list(zip(*iterables)))):
                    jobs.append(
                        HTCondorJob(cluster_id, i, self.schedd, self.htcondor_dir)
                    )

                return jobs

            def _store_job_params(self, job_idx: int, func, args):
                """Store job parameters for HTCondor process to load."""
                import pickle

                call_data = {"func": func, "args": args}

                pickle_path = self.htcondor_dir / f"job_{job_idx}.pkl"
                with open(pickle_path, "wb") as f:
                    pickle.dump(call_data, f)

            def _create_submit_description(self, num_jobs: int):
                """Create HTCondor submit description for job array."""
                # Create the job runner script that will be executed by HTCondor
                runner_script = self._create_job_runner_script()

                # Build the submit description
                submit_dict = {}

                # Executable - use custom executable if provided, otherwise Python
                if self.params.get("executable"):
                    submit_dict["executable"] = self.params["executable"]
                    # If custom executable, pass the runner script as argument
                    submit_dict["arguments"] = f"{runner_script} $(Process)"
                else:
                    submit_dict["executable"] = sys.executable
                    submit_dict["arguments"] = f"{runner_script} $(Process)"

                # Output files with HTCondor variable substitution
                output_dir = self.params.get("output_dir", str(self.htcondor_dir))
                submit_dict["output"] = self.params.get(
                    "output", f"{output_dir}/$(Cluster)_$(Process).out"
                )
                submit_dict["error"] = self.params.get(
                    "error", f"{output_dir}/$(Cluster)_$(Process).err"
                )
                submit_dict["log"] = self.params.get(
                    "log", f"{output_dir}/$(Cluster)_$(Process).log"
                )

                # Resource requests
                submit_dict["request_memory"] = self.params.get(
                    "request_memory", "4000"
                )
                submit_dict["request_cpus"] = self.params.get("request_cpus", "1")
                submit_dict["request_gpus"] = self.params.get("request_gpus", "0")

                # File transfer
                submit_dict["should_transfer_files"] = "YES"
                submit_dict["when_to_transfer_output"] = "ON_EXIT"
                submit_dict["transfer_input_files"] = (
                    f"{runner_script},{self.htcondor_dir}"
                )
                submit_dict["getenv"] = "True"

                # Optional parameters
                if self.params.get("requirements"):
                    submit_dict["requirements"] = self.params["requirements"]

                if self.params.get("MaxTime"):
                    submit_dict["MaxTime"] = str(self.params["MaxTime"])
                    # Add periodic remove based on MaxTime
                    submit_dict["periodic_remove"] = (
                        f"(JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))"
                    )

                # Any additional custom parameters
                for key, value in self.params.items():
                    if key not in [
                        "executable",
                        "output",
                        "error",
                        "log",
                        "request_memory",
                        "request_cpus",
                        "request_gpus",
                        "requirements",
                        "MaxTime",
                        "output_dir",
                        "htcondor_folder",
                    ]:
                        submit_dict[key] = str(value)

                return htcondor.Submit(submit_dict)

            def _create_job_runner_script(self):
                """Create the main job runner script that HTCondor will execute."""
                runner_script = self.htcondor_dir / "hydra_job_runner.py"

                script_content = f"""#!/usr/bin/env python3
import sys
import pickle
import os
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: hydra_job_runner.py <process_id>")
        sys.exit(1)
    
    process_id = int(sys.argv[1])
    
    # Set HTCondor environment variables for Hydra
    cluster_id = os.environ.get("_CONDOR_CLUSTER_ID", "unknown")
    os.environ["HYDRA_JOB_ID"] = f"{{cluster_id}}.{{process_id}}"
    os.environ["HYDRA_JOB_NUM"] = str(process_id)
    
    # Load job parameters
    htcondor_dir = Path("{self.htcondor_dir}")
    job_params_file = htcondor_dir / f"job_{{process_id}}.pkl"
    
    try:
        with open(job_params_file, "rb") as f:
            call_data = pickle.load(f)
            
        func = call_data["func"]
        args = call_data["args"]
        
        # Execute the job
        result = func(*args)
        
        # Save result
        result_file = htcondor_dir / f"job_{{process_id}}_result.pkl"
        with open(result_file, "wb") as f:
            pickle.dump(result, f)
            
        print(f"Job {{cluster_id}}.{{process_id}} completed successfully")
        
    except Exception as e:
        print(f"Job {{cluster_id}}.{{process_id}} failed: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

                with open(runner_script, "w") as f:
                    f.write(script_content)

                os.chmod(runner_script, 0o755)
                return runner_script

        class HTCondorJob:
            def __init__(
                self, cluster_id: int, job_idx: int, schedd, htcondor_dir: Path
            ):
                self.cluster_id = cluster_id
                self.job_idx = job_idx
                self.schedd = schedd
                self.htcondor_dir = htcondor_dir

            def result(self):
                """Wait for job completion and return result."""
                import time

                # Wait for job to complete
                while True:
                    jobs = self.schedd.query(
                        f"ClusterId == {self.cluster_id}", ["JobStatus", "ExitCode"]
                    )

                    if not jobs:
                        break

                    job = jobs[0]
                    if job["JobStatus"] == htcondor.JobStatus.COMPLETED:
                        # Load result
                        result_path = (
                            self.htcondor_dir / f"job_{self.job_idx}_result.pkl"
                        )
                        if result_path.exists():
                            with open(result_path, "rb") as f:
                                return pickle.load(f)
                        else:
                            return JobReturn()
                    elif job["JobStatus"] in [
                        htcondor.JobStatus.REMOVED,
                        htcondor.JobStatus.HELD,
                    ]:
                        return JobReturn()

                    time.sleep(5)

                return JobReturn()

        return HTCondorExecutor(htcondor_dir, self.params)
