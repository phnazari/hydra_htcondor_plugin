# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pytest import mark

from hydra.core.plugins import Plugins
from hydra.plugins.launcher import Launcher
from hydra.test_utils.launcher_common_tests import (
    IntegrationTestSuite,
    LauncherTestSuite,
)


from hydra_plugins.hydra_htcondor_launcher.htcondor_launcher import HTCondorLauncher


def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking for Launchers
    assert HTCondorLauncher.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Launcher)
    ]


@mark.parametrize("launcher_name, overrides", [("htcondor_launcher", [])])
class TestHTCondorLauncher(LauncherTestSuite):
    """
    Run the Launcher test suite on this launcher.
    Note that hydra/launcher/htcondor.yaml should be provided by this launcher.
    """

    pass


@mark.parametrize(
    "task_launcher_cfg, extra_flags",
    [({}, ["-m", "hydra/launcher=htcondor_launcher"])],
)
class TestHTCondorLauncherIntegration(IntegrationTestSuite):
    """
    Run this launcher through the integration test suite.
    """

    pass
