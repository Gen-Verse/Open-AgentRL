import os
import sys
import subprocess
import shlex
import json
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path
import shutil
import re

def run_local(cmd: str, check: bool = True) -> None:
    subprocess.run(f"bash -lc {shlex.quote(cmd)}", shell=True, check=check)

def run_local_async(cmd: str) -> subprocess.Popen:
    return subprocess.Popen(f"bash -lc {shlex.quote(cmd)}", shell=True)

def run_remote(host: str, cmd: str, check: bool = True) -> None:
    ssh_cmd = f'ssh root@{host} "bash -lc {shlex.quote(cmd)}"'
    subprocess.run(ssh_cmd, shell=True, check=check)

def run_remote_async(host: str, cmd: str) -> subprocess.Popen:
    ssh_cmd = f'ssh root@{host} "bash -lc {shlex.quote(cmd)}"'
    return subprocess.Popen(ssh_cmd, shell=True)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)

def begin_with(file_name: str):
    with open(file_name, "w"):
        pass


def make_init_bash(cfg) -> str:
    sc = cfg.system
    download_proxy  = getattr(sc, "download_proxy", None)
    hf_home         = getattr(sc, "HF_HOME", None)
    envs_dir        = getattr(sc, "envs_dir", None)
    additional_command = getattr(sc, "additional_command", None)

    ve_ACCESS_KEY_ID        = getattr(sc, "VOLCENGINE_ACCESS_KEY_ID", None)
    ve_SECRET_ACCESS_KEY    = getattr(sc, "VOLCENGINE_SECRET_ACCESS_KEY", None)
    ve_REGION               = getattr(sc, "VOLCENGINE_REGION", None)
    ve_IMAGE_ID             = getattr(sc, "VOLCENGINE_IMAGE_ID", None)
    ve_INSTANCE_TYPE        = getattr(sc, "VOLCENGINE_INSTANCE_TYPE", None)
    ve_SUBNET_ID            = getattr(sc, "VOLCENGINE_SUBNET_ID", None)
    ve_SECURITY_GROUP_ID    = getattr(sc, "VOLCENGINE_SECURITY_GROUP_ID", None)
    ve_ZONE_ID              = getattr(sc, "VOLCENGINE_ZONE_ID", None)
    ve_DEFAULT_PASSWORD     = getattr(sc, "VOLCENGINE_DEFAULT_PASSWORD", None)

    def write_export(lines, file, k, v):
        if v is not None:
            lines.append(f"echo 'export {k}={v}' >> {file}")

    from omegaconf import ListConfig
    if isinstance(ve_INSTANCE_TYPE, (list, ListConfig)):
        ve_INSTANCE_TYPE_str = ",".join(str(x) for x in ve_INSTANCE_TYPE)
    else:
        ve_INSTANCE_TYPE_str = ve_INSTANCE_TYPE

    lines = []
    lines.append("set -e")

    for f in ("~/.bashrc", "~/.bash_profile"):
        write_export(lines, f, "download_proxy",  download_proxy)
        write_export(lines, f, "HF_HOME",         hf_home)
        if hf_home is not None:
            lines.append(f"echo 'export TRANSFORMERS_CACHE={hf_home}' >> {f}")
        write_export(lines, f, "VOLCENGINE_ACCESS_KEY_ID",     ve_ACCESS_KEY_ID)
        write_export(lines, f, "VOLCENGINE_SECRET_ACCESS_KEY", ve_SECRET_ACCESS_KEY)
        write_export(lines, f, "VOLCENGINE_REGION",            ve_REGION)
        write_export(lines, f, "VOLCENGINE_IMAGE_ID",          ve_IMAGE_ID)
        write_export(lines, f, "VOLCENGINE_INSTANCE_TYPE",     ve_INSTANCE_TYPE_str)
        write_export(lines, f, "VOLCENGINE_SUBNET_ID",         ve_SUBNET_ID)
        write_export(lines, f, "VOLCENGINE_SECURITY_GROUP_ID", ve_SECURITY_GROUP_ID)
        write_export(lines, f, "VOLCENGINE_ZONE_ID",           ve_ZONE_ID)
        write_export(lines, f, "VOLCENGINE_DEFAULT_PASSWORD",  ve_DEFAULT_PASSWORD)

    if additional_command is not None:
        lines.append(additional_command)
    lines.append("")

    if envs_dir is not None:
        lines.append(f"conda config --append envs_dirs {envs_dir} || true")
        lines.append("")

    lines.append("echo 'source ~/.bashrc' >> ~/.bash_profile")
    lines.append("")

    return "\n".join(lines)


def cleanup_orphan_instances_and_logs(cfg):

    import os
    import shutil

    try:
        import volcenginesdkcore
        import volcenginesdkecs.models as ecs_models
        from volcenginesdkecs.api import ECSApi
        from volcenginesdkcore.rest import ApiException
    except ImportError as e:
        print(f"[cleanup] Volcengine SDK not available: {e}")
        print("[cleanup] Please install volcengine SDK or cleanup manually.")
        return

    ak = getattr(getattr(cfg, "system", object()), "VOLCENGINE_ACCESS_KEY_ID", None) or os.getenv("VOLCENGINE_ACCESS_KEY_ID")
    sk = getattr(getattr(cfg, "system", object()), "VOLCENGINE_SECRET_ACCESS_KEY", None) or os.getenv("VOLCENGINE_SECRET_ACCESS_KEY")
    region = getattr(getattr(cfg, "system", object()), "VOLCENGINE_REGION", None) or os.getenv("VOLCENGINE_REGION")

    if not ak or not sk or not region:
        print("[cleanup] Missing VOLCENGINE credentials (ak/sk/region). Skip remote cleanup.")
        return

    try:
        configuration = volcenginesdkcore.Configuration()
        configuration.ak = ak
        configuration.sk = sk
        configuration.region = region
        configuration.client_side_validation = True
        volcenginesdkcore.Configuration.set_default(configuration)

        ecs_client = ECSApi()
    except Exception as e:
        print(f"[cleanup] Error while initializing Volcengine client: {e}")
        return

    print("[cleanup] Listing instances in region:", region)

    to_delete = []

    try:

        req = ecs_models.DescribeInstancesRequest()
        resp = ecs_client.describe_instances(req)

        instances = getattr(resp, "instances", []) or []

        for inst in instances:

            name = getattr(inst, "instance_name", None) or getattr(inst, "instanceName", None) or ""
            iid  = getattr(inst, "instance_id", None)  or getattr(inst, "instanceId", None)  or None
            status = getattr(inst, "status", None)

            if not iid:
                continue

            if isinstance(name, str) and name.startswith(f"{cfg.experiment.project}-"):
                print(f"[cleanup] Mark for delete: {iid} (name={name}, status={status})")
                to_delete.append(iid)

    except ApiException as e:
        print(f"[cleanup] ApiException when listing instances: {e}")
    except Exception as e:
        print(f"[cleanup] Unexpected error when listing instances: {e}")

    if not to_delete:
        print(f"[cleanup] No instances with name starting with {cfg.experiment.project}- found.")
    else:
        print(f"[cleanup] Will delete {len(to_delete)} instances: {to_delete}")

        for iid in to_delete:
            try:
                req = ecs_models.DeleteInstanceRequest(instance_id=iid)
                resp = ecs_client.delete_instance(req)
                print(f"[cleanup] delete_instance({iid}) OK: {resp}")
            except ApiException as e:
                print(f"[cleanup] ApiException when deleting {iid}: {e}")
            except Exception as e:
                print(f"[cleanup] Unexpected error when deleting {iid}: {e}")





if __name__ == "__main__":
    cfg = get_config()

    BASE_DIR = cfg.system.rl_base_dir
    env_name = cfg.system.env_name
    project = cfg.experiment.project
    
    project_dir = Path(BASE_DIR) / project
    (project_dir / "temp_data").mkdir(parents=True, exist_ok=True)
    (project_dir / "results").mkdir(parents=True, exist_ok=True)

    def env_prefix(env) -> str:
        return (
            "source ~/.bashrc && "
            f"source activate {env} && "
        )


    def init_hosts(worker_hosts, init_bash: str):

        procs = []
        for idx, host in enumerate(worker_hosts):
            if idx == 0:
                procs.append(run_local_async(init_bash))
            else:
                procs.append(run_remote_async(host, init_bash))
        for p in procs:
            p.wait()

    def start_serve(worker_hosts, cfg, type, model, policy_model_type):

        osw = "OSWorld-main"
        if type == "train":
            per   = int(cfg.rollout.policy.num_gpu_per_model)
        else:
            per   = int(cfg.rollout.policy_evaluation.num_gpu_per_model)
        
        if policy_model_type == "qwen3vl":
            script_name = "start_8gpus_qwen3vl.sh"
        elif policy_model_type == "uitars15":
            script_name = "start_8gpus_uitars15.sh"
        else:
            script_name = "start_8gpus_opencua.sh"

        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR} && "
                f"cd {osw} && "
                f"chmod +x {script_name} && "
                f"MODEL={shlex.quote(str(model))} "
                f"NUM_GPU_PER_MODEL={per} "
                f"./{script_name}"
            )
            full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()

    def run_sample(worker_hosts, cfg, type, model, policy_model_type):

        start_serve(worker_hosts, cfg, type, model, policy_model_type)

        osw = "OSWorld-main"
        num_node = int(cfg.experiment.num_node)

        if type == "train":
            max_steps = str(cfg.rollout.policy.max_steps)
            num_env = str(cfg.rollout.num_envs)
            num_rollout_per_trial = str(cfg.rollout.policy.num_rollout_per_trial)
            temperature = str(cfg.rollout.policy.temperature)
            num_trial = str(cfg.rollout.policy.num_trial)
            environment_data_dir = cfg.dataset.train.environment_data_dir
            domain = cfg.dataset.train.domain
            example = cfg.dataset.train.example
        else:
            max_steps = str(cfg.rollout.policy_evaluation.max_steps)
            num_env = str(cfg.rollout.num_envs)
            num_rollout_per_trial = str(cfg.rollout.policy_evaluation.num_rollout_per_trial)
            temperature = str(cfg.rollout.policy_evaluation.temperature)
            num_trial = str(cfg.rollout.policy_evaluation.num_trial)
            environment_data_dir = cfg.dataset.evaluation.environment_data_dir
            domain = cfg.dataset.evaluation.domain
            example = cfg.dataset.evaluation.example

        if policy_model_type == "qwen3vl":
            script_name = "rl_rollout_local_qwen3vl.py"
        elif policy_model_type == "uitars15":
            script_name = "rl_rollout_local_uitars15.py"
        else:
            script_name = "rl_rollout_local_opencua.py"
        
        if type == "train":
            coevolve_environment = "TRUE"
        else:
            coevolve_environment = "FALSE"
        
        procs = []
        for idx, host in enumerate(worker_hosts):
            args = [
                "--headless",
                "--observation_type", "screenshot",
                "--model",            shlex.quote(str(model)),
                "--result_dir",       shlex.quote(f"{cfg.rollout.result_dir}/{cfg.experiment.project}"),
                "--test_all_meta_path", f"evaluation_examples/{environment_data_dir}.json",
                "--max_steps",        max_steps,
                "--num_envs",         num_env,
                "--region",           shlex.quote(str(cfg.system.region)),
                "--coordinate_type",  str(cfg.rollout.coordinate_type),
                "--num_rollout_per_trial", num_rollout_per_trial,
                "--domain",           shlex.quote(str(domain)),
                "--example",          shlex.quote(str(example)),
                "--temperature",      temperature,
                "--use_old_sys_prompt",
                "--num_node",         str(num_node),
                "--node_index",       str(idx),
                "--rollout_type",     shlex.quote(str(type)),
                "--num_trial",        num_trial,
                "--save_example_json",str(f"{cfg.system.rl_base_dir}/{cfg.experiment.project}/temp_data/example_node_{idx}.json"),
                "--action_space",     str(cfg.rollout.action_space),
                "--project",          str(cfg.experiment.project),
                "--observation_type", str(cfg.rollout.observation_type),
                "--coevolveenv",      str(coevolve_environment),
                "--current_step",     str(cfg.experiment.current_epoch)
            ]
            if bool(cfg.experiment.if_rerun):
                args.append("--rerun")

            body = (
                f"cd {BASE_DIR} && " +
                f"cd {osw} && " +
                f"python {script_name} " +
                " ".join(args)
            )
            full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()
        
        stop_serve(worker_hosts)

    def stop_serve(worker_hosts):

        osw = "OSWorld-main"
        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR} && "
                f"cd {osw} && "
                "chmod +x stop_8gpus.sh || true && "
                "./stop_8gpus.sh || true"
            )
            full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()
    

    def run_reward_sample(worker_hosts, cfg, model, reward_model):

        osw = "sample"
        model_tag = re.sub(r'[\\/]+', '_', model).lstrip('_')
        gpu_groups_py = OmegaConf.to_container(cfg.rollout.reward.gpu_groups, resolve=True)
        gpu_groups_str = json.dumps(gpu_groups_py, separators=(",", ":"))

        procs = []
        for idx, host in enumerate(worker_hosts):
            args = [
                "--root-dir",               shlex.quote(f"{cfg.rollout.result_dir}/{cfg.experiment.project}/{cfg.rollout.action_space}/{cfg.rollout.observation_type}/{model_tag}"),
                "--examples-json",          shlex.quote(f"{cfg.system.rl_base_dir}/{cfg.experiment.project}/temp_data/example_node_{idx}.json"),
                "--output-json",            shlex.quote(f"{cfg.system.rl_base_dir}/{cfg.experiment.project}/temp_data/reward_rollout_results_node_{idx}.json"),
                "--model",                  shlex.quote(str(reward_model)),
                "--num-rollout-per-query",  str(cfg.rollout.reward.num_rollout_per_query),
                "--download_proxy",         str(cfg.system.download_proxy),
                "--max-tokens",             str(cfg.rollout.reward.max_tokens), 
                "--temperature",            str(cfg.rollout.reward.temperature),
                "--gpu-groups",             shlex.quote(gpu_groups_str)
            ]

            body = (
                f"cd {BASE_DIR} && " +
                f"cd {osw} && " +
                f"python osworld_reward_rollout.py " +
                " ".join(args)
            )
            full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()
    
    
    def reward(step, cfg, type, model):
        project = cfg.experiment.project
        num_node = int(cfg.experiment.num_node)
        model_tag = re.sub(r'[\\/]+', '_', model).lstrip('_')
        root_dir = (
            f"{cfg.rollout.result_dir}/"
            f"{project}/"
            f"{cfg.rollout.action_space}/"
            f"{cfg.rollout.observation_type}/"
            f"{model_tag}"
        )
        merge_dir = f"{BASE_DIR}/{project}/temp_data"
        full_cmd = env_prefix(env_name) + (
            f"cd {BASE_DIR}/reward && "
            f"python osworld_rl_reward.py "
            f"--root-dir {shlex.quote(root_dir)} "
            f"--num-nodes {num_node} "
            f"--type {type} "
            f"--step {step} "
            f"--merge-dir {shlex.quote(merge_dir)}"
        )

        run_local(full_cmd)
    

    def process_shards(worker_hosts, epoch, cfg, model_type, target):

        project = cfg.experiment.project
        num_nodes = len(worker_hosts)

        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR} && "
                "export DS_SKIP_CUDA_CHECK=1 && "
                "python -m train.osworld_vlm_preprocess_shards "
                f"config=configs/{project}.yaml "
                f"training.target={target} "
                f"experiment.current_epoch={epoch} "
                f"model_type={model_type} "
                "cast_pixel_fp16=1 "
                "pad_to_fixed=1 "
                f"dataset.num_nodes={num_nodes} "
                f"dataset.node_rank={idx} "
            )

            full_cmd = env_prefix(env_name) + body

            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))

            print(f"[DISPATCH] preprocess node {idx} → {host}")

        for p in procs:
            p.wait()

        print("All preprocess nodes finished.")
    
    def train_agg(cfg, model, model_type, epoch, target):
        project = cfg.experiment.project
        num_node = int(cfg.experiment.num_node)
        model_tag = re.sub(r'[\\/]+', '_', model).lstrip('_')
        merge_dir = f"{BASE_DIR}/{project}/temp_data"
        full_cmd = env_prefix(env_name) + (
            f"cd {BASE_DIR} && "
            f"python -m train.osworld_vlm_merge_preproc_shards "
            f"config=configs/{project}.yaml "
            f'training.target={target} '
            f'model_type={model_type} '
            f"experiment.current_epoch={epoch}"
        )
        run_local(full_cmd)

    def train(worker_hosts, epoch, cfg, model_type, target = None):

        process_shards(worker_hosts, epoch, cfg, model_type, target)

        train_agg(cfg, model, model_type, epoch, target)
        
        project = cfg.experiment.project
        ds_file = cfg.experiment.deepspeed_file
        num_nodes = len(worker_hosts)
        master_ip = os.environ["MLP_WORKER_0_HOST"]
        master_port = os.environ["MLP_WORKER_0_PORT"]
        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR} && "
                "export DS_SKIP_CUDA_CHECK=1 && "
                "accelerate launch "
                f"--num_machines {num_nodes} "
                f"--machine_rank {idx} "
                f"--main_process_ip {master_ip} "
                f"--main_process_port {master_port} "
                f"--config_file accelerate_configs/{ds_file}.yaml "
                f"train/osworld_train.py "
                f"config=configs/{project}.yaml "
                f'training.target={target} '
                f'model_type={model_type} '
                f"experiment.current_epoch={epoch}"
            )
            if target == "policy":
                full_cmd = env_prefix(env_name) + body
            else:
                full_cmd = env_prefix(env_name) + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
            print(f"[DISPATCH] train node {idx} → {host}")
        for p in procs:
            p.wait()
        print("All train nodes finished.")



    def clear_dir(dir_to_clean):
        p = Path(dir_to_clean)
        p.mkdir(parents=True, exist_ok=True)  

        for child in p.iterdir():             
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                shutil.rmtree(child)
    
    def clear_results_if_start_from_scratch(cfg):

        if not getattr(cfg.experiment, "start_from_scratch", False):
            return

        project = cfg.experiment.project
        results_dir = f"{BASE_DIR}/{project}/results"
        clear_dir(f"{BASE_DIR}/{project}/temp_data")
        for name in ("rl-results.txt", "eval-results.txt"):
            path = os.path.join(results_dir, name)
            if os.path.exists(path):
                begin_with(path)  

    def clear_copy_coevolve_train_files(cfg):
        src_sub_dir  = "new_examples"
        dst_sub_dir  = f"{cfg.experiment.project}/train"
        temp_sub_dir = f"{cfg.experiment.project}/temp"

        test_config_base_dir = Path(BASE_DIR) / "OSWorld-main" / "evaluation_examples"

        src_path  = test_config_base_dir / src_sub_dir          
        dst_path  = test_config_base_dir / dst_sub_dir          
        temp_path = test_config_base_dir / temp_sub_dir         

        if not src_path.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {src_path}")

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            shutil.rmtree(dst_path)

        shutil.copytree(src_path, dst_path)   
        clear_dir(str(temp_path))             
    
    def env_eoevolve(cfg):
        project = cfg.experiment.project
        script_name = "osworld_env_rollout.py"
        full_cmd = env_prefix(env_name) + (
            f"cd {BASE_DIR}/sample && "
            f"python {script_name} "
            f"config=../configs/{project}.yaml"
        )
        run_local(full_cmd)

    INIT_BASH = make_init_bash(cfg)

    num_node = int(cfg.experiment.num_node)
    worker_hosts = [os.environ[f"MLP_WORKER_{i}_HOST"] for i in range(num_node)]

    import time
    time.sleep(30)

    init_hosts(worker_hosts, INIT_BASH)

    import time
    time.sleep(10)

    step = cfg.experiment.current_epoch
    
    clear_results_if_start_from_scratch(cfg)

    if cfg.experiment.start_from_scratch:
        clear_copy_coevolve_train_files(cfg)

    while step <= cfg.experiment.total_step:

        if step == 1:
            model = cfg.model.policy_model
            reward_model = cfg.model.reward_model
        else:
            model = f"{cfg.system.rl_base_dir}/{cfg.experiment.project}/ckpt/{cfg.model.optimized_name}"
            reward_model = f"{cfg.system.rl_base_dir}/{cfg.experiment.project}/ckpt/{cfg.model.optimized_reward_name}"

        clear_dir(f"{cfg.rollout.result_dir}/{cfg.experiment.project}")
        run_sample(worker_hosts, cfg, "train", model, cfg.model.policy_model_type)
        for _ in range(20):
            cleanup_orphan_instances_and_logs(cfg)

        run_reward_sample(worker_hosts, cfg, model, reward_model)

        reward(step, cfg, "train", model)
        
        if cfg.experiment.start_from_scratch:
            env_eoevolve(cfg)

        train(worker_hosts, step, cfg, cfg.model.policy_model_type, target = "policy")
        train(worker_hosts, step, cfg, cfg.model.reward_model_type, target = "reward")

        if step % cfg.experiment.eval_every == 0:
            clear_dir(f"{cfg.rollout.result_dir}/{cfg.experiment.project}")
            run_sample(worker_hosts, cfg, "evaluation", model, cfg.model.policy_model_type)
            for _ in range(20):
                cleanup_orphan_instances_and_logs(cfg)

            reward(step, cfg, "evaluation", model)
        
        step += 1
        







