import json
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
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

    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + config.dataset.environment_type + "-" + config.dataset.alfworld_data_type
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        data = json.load(f)

    response_list = []
    max_prompt_list = []
    num_all = 0
    num_success = 0
    sum_success = 0
    for i in range(len(data)):
        for j in range(len(data[i]["prompt"])):
            max_prompt_list.append(data[i]["prompt"][j][-1])
            response_list = response_list + data[i]["response"][j]
        for j in range(len(data[i]["if_success"])):
            num_all += 1
            if data[i]["if_success"][j] == 1:
                num_success += 1
                sum_success += data[i]["success_steps"][j]
            else:
                sum_success += config.rollout.max_interaction_step
    
    acc =  num_success / num_all
    avg_step = sum_success / num_all
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def get_lengths(strings, tokenizer, type):
        response_length = [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]
        if type == "mean":
            return sum(response_length)/len(response_length)
        elif type == "max":
            return max(response_length)

    avg_response_length = get_lengths(response_list, tokenizer, "mean")
    max_prompt_length = get_lengths(max_prompt_list, tokenizer, "max")
        
        
        
    
    import os
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")

        save_and_print(f"acc: {acc}   avg step: {avg_step}   avg response length: {avg_response_length}   max_prompt_length: {max_prompt_length}")
