import os
import sys
import subprocess
import shlex
import re
import shutil
from omegaconf import DictConfig, ListConfig, OmegaConf


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


from pathlib import Path

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

    def start_serve(worker_hosts, cfg, model_type):

        osw = "OSWorld-main"
        model = cfg.model
        per   = int(cfg.rollout.num_gpu_per_model)

        if model_type == "qwen3vl":
            script_name = "start_8gpus_qwen3vl.sh"
        elif model_type == "uitars15":
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

    def run_sample(worker_hosts, cfg, model_type):

        osw = "OSWorld-main"
        num_node = int(cfg.experiment.num_node)

        start_serve(worker_hosts, cfg, model_type)

        procs = []
        for idx, host in enumerate(worker_hosts):
            args = [
                "--headless",
                "--observation_type", "screenshot",
                "--model",            shlex.quote(str(cfg.model)),
                "--result_dir",       shlex.quote(f"{cfg.rollout.result_dir}/{cfg.experiment.project}"),
                "--test_all_meta_path", f"evaluation_examples/{cfg.dataset.environment_data_dir}.json",
                "--max_steps",        str(cfg.rollout.max_steps),
                "--num_envs",         str(cfg.rollout.num_envs),
                "--region",           shlex.quote(str(cfg.system.region)),
                "--coordinate_type",  str(cfg.rollout.coordinate_type),
                "--num_rollout_per_trial", str(cfg.rollout.num_rollout_per_trial),
                "--domain",           shlex.quote(str(cfg.dataset.domain)),
                "--example",          shlex.quote(str(cfg.dataset.example)),
                "--temperature",      str(cfg.rollout.temperature),
                "--use_old_sys_prompt",
                "--num_node",         str(num_node),
                "--node_index",       str(idx),
                "--action_space",     str(cfg.rollout.action_space),
                "--project",          str(cfg.experiment.project),
                "--observation_type", str(cfg.rollout.observation_type)
            ]
            if bool(cfg.experiment.if_rerun):
                args.append("--rerun")

            if model_type == "qwen3vl":
                script_name = "sample_local_qwen3vl.py"
            elif model_type == "uitars15":
                script_name = "sample_local_uitars15.py"
            else:
                script_name = "sample_local_opencua.py"
            
            body = (
                f"cd {BASE_DIR} && " +
                f"cd {osw} && " +
                f"{shlex.quote(sys.executable)} {script_name} " +
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

    def reward(cfg, model):
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
            f"python osworld_reward.py "
            f"--root-dir {shlex.quote(root_dir)} "
            f"--num-nodes {num_node} "
            f"--merge-dir {shlex.quote(merge_dir)}"
        )

        run_local(full_cmd)

    def clear_results_if_start_from_scratch(cfg):

        if not getattr(cfg.experiment, "start_from_scratch", False):
            return

        project = cfg.experiment.project
        results_dir = f"{BASE_DIR}/{project}/results"
        for name in ("eval-results.txt"):
            path = os.path.join(results_dir, name)
            if os.path.exists(path):
                begin_with(path)  

    def clear_dir(dir_to_clean):
        p = Path(dir_to_clean)
        p.mkdir(parents=True, exist_ok=True)  

        for child in p.iterdir():             
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                shutil.rmtree(child)
    
    # ========================= main：只做分发 =========================
    INIT_BASH = make_init_bash(cfg)

    clear_results_if_start_from_scratch(cfg)

    # 按你的风格：worker_hosts 在 main 里创建，然后传给各个函数
    num_node = int(cfg.experiment.num_node)
    worker_hosts = [os.environ[f"MLP_WORKER_{i}_HOST"] for i in range(num_node)]

    import time
    time.sleep(30)

    model_type = cfg.model_type

    # 一次 init（本机 + 所有远端）
    init_hosts(worker_hosts, INIT_BASH)

    import time
    time.sleep(10)

    # 多机：start → run → stop
    clear_dir(f"{cfg.rollout.result_dir}/{cfg.experiment.project}")
    run_sample(worker_hosts, cfg, model_type)

    for _ in range(20):
        cleanup_orphan_instances_and_logs(cfg)

    reward(cfg, cfg.model)
    
