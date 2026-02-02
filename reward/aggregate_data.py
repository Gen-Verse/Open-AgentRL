import json
import math_utils
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
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
    pretrained_model = config.model
    if config.dataset.environment_type == "alfworld":
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + config.dataset.environment_type + "-" + config.dataset.alfworld_data_type
    else:
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + config.dataset.eval_dataset


    num_slice = config.experiment.num_node
    final_data = []
    for i in range(num_slice):
        file_name_sub = "../" + project_name + f"/temp_data/outputs-{i}-" + outputs_name + ".json"
        with open(file_name_sub, 'r') as f:
            data = json.load(f)
        os.remove(file_name_sub)
        final_data = final_data + data
    
    file_name = "../" + project_name + f"/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
