import os
import sys
import subprocess
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":
    config = get_config()

    project_name = config.experiment.project
    env_type = config.dataset.environment_type

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")
        
    def sample():
        cprint(f"This is sampling.", color = "green")
        if env_type == "alfworld":
            script_name = "alfworld_sample.py"
        subprocess.run(
            f'python {script_name} '
            f'config=../configs/{project_name}.yaml ',
            shell=True,
            cwd='sample',
            check=True,
        )
        
    
    def reward():
        cprint(f"This is the rewarding.", color = "green")
        if env_type == "alfworld":
            script_name = "alfworld_reward.py"
        subprocess.run(
            f'python {script_name} '
            f'config=../configs/{project_name}.yaml ',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    os.makedirs(f"{project_name}/results", exist_ok=True)
    
    sample()
    
    reward()




