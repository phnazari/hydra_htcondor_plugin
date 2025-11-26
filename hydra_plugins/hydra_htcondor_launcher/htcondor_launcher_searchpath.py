# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins


class HTCondorLauncherSearchPathPlugin(SearchPathPlugin):
    """Adds the HTCondor launcher config to Hydra's search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Add the config directory from this package to the search path
        search_path.append(
            provider="htcondor_launcher",
            path="pkg://hydra_plugins.hydra_htcondor_launcher.conf",
        )

