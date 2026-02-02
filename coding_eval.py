import os
import subprocess
import shlex
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
    http_proxy  = sc.HTTP_PROXY
    https_proxy = sc.HTTP_PROXY
    hf_home     = sc.HF_HOME
    envs_dir    = sc.envs_dir

    lines = []
    lines.append("set -e")
    if http_proxy is not None:
        lines.append(f"echo 'export HTTP_PROXY={http_proxy}' >> ~/.bashrc")
    if https_proxy is not None:
        lines.append(f"echo 'export HTTPS_PROXY={https_proxy}' >> ~/.bashrc")
    if hf_home is not None:
        lines.append(f"echo 'export HF_HOME={hf_home}' >> ~/.bashrc")
    lines.append("") 

    if envs_dir is not None:
        lines.append(f"conda config --append envs_dirs {envs_dir} || true")
        lines.append("") 

    lines.append("echo 'source ~/.bashrc' >> ~/.bash_profile")
    lines.append("")

    return "\n".join(lines)








if __name__ == "__main__":


    def init_node(host: str):
        run_remote(host, INIT_BASH, check=False)

    def init_hosts(worker_hosts):
        for h in worker_hosts:
            if h is None:
                continue
            init_node(h)


    def env_prefix() -> str:
        return (
            "source ~/.bashrc && "
            f"source activate {env_name} && "
        )


    def policy_sample(worker_hosts, epoch, cfg, type):
        project = cfg.experiment.project
        procs = []
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR}/sample && "
                f"python llm_policy_rollout.py "
                f"config=../configs/{project}.yaml "
                f"experiment.current_epoch={epoch} "
                f"experiment.function={type} "
                f"experiment.node_index={idx}"
            )
            full_cmd = env_prefix() + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()
    
    def reward_sample(worker_hosts, epoch, cfg, type):
        project = cfg.experiment.project
        procs = []
        script_name = "llm_reward_rollout.py"
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR}/sample && "
                f"PYTHONUNBUFFERED=1 python -u {script_name} "
                f"config=../configs/{project}.yaml "
                f"experiment.current_epoch={epoch} "
                f"experiment.function={type} "
                f"experiment.node_index={idx}"
            )
            full_cmd = env_prefix() + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()



    def aggregate(epoch, cfg, type):
        project = cfg.experiment.project
        full_cmd = env_prefix() + (
            f"cd {BASE_DIR}/reward && "
            f"python rl_aggregate_data.py "
            f"config=../configs/{project}.yaml "
            f"experiment.function={type} "
            f"experiment.current_epoch={epoch}"
        )
        run_local(full_cmd)

    
    def execute(worker_hosts, epoch, cfg, type):
        project = cfg.experiment.project
        procs = []
        script_name = "execute.py"
        for idx, host in enumerate(worker_hosts):
            body = (
                f"cd {BASE_DIR}/reward && "
                f"python {script_name} "
                f"config=../configs/{project}.yaml "
                f"experiment.current_epoch={epoch} "
                f"experiment.function={type} "
                f"experiment.node_index={idx}"
            )
            full_cmd = env_prefix() + body
            if idx == 0:
                procs.append(run_local_async(full_cmd))
            else:
                procs.append(run_remote_async(host, full_cmd))
        for p in procs:
            p.wait()
    
    def reward(epoch, cfg, type):
        project = cfg.experiment.project
        script_name = "llm_rl_reward.py"
        full_cmd = env_prefix() + (
            f"cd {BASE_DIR}/reward && "
            f"python {script_name} "
            f"config=../configs/{project}.yaml "
            f"experiment.function={type} "
            f"experiment.current_epoch={epoch}"
        )
        run_local(full_cmd)


    cfg = get_config()
    INIT_BASH = make_init_bash(cfg)
    BASE_DIR = cfg.system.rl_base_dir
    env_name = cfg.system.env_name
    project = cfg.experiment.project
    num_node = cfg.experiment.num_node
    #worker_hosts = [os.environ[f"MLP_WORKER_{i}_HOST"] for i in range(num_node)]
    if num_node <= 1:
        worker_hosts = [None]  # rank0 local placeholder
    else:
        worker_hosts = [os.environ[f"MLP_WORKER_{i}_HOST"] for i in range(num_node)]

    import time
    time.sleep(30)

    init_hosts(worker_hosts)

    import time
    time.sleep(10)

    if cfg.experiment.start_from_scratch:
        os.makedirs(f"{project}/results", exist_ok=True)
        model_eval = cfg.model.policy_model
        path = (
            f"{project}/results/results-eval-"
            f"{model_eval.replace('/', '.')}-"
            f"{cfg.dataset.eval_dataset}.txt"
        )
        begin_with(path)
        import shutil

        def clear_dir(out_dir):
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
        clear_dir(f"{project}/temp_data")
    
    
    epoch = 1

    policy_sample(worker_hosts, epoch, cfg, "evaluation")
    reward_sample(worker_hosts, epoch, cfg, "evaluation")
    execute(worker_hosts, epoch, cfg, "evaluation")
    aggregate(epoch, cfg, "evaluation")
    reward(epoch, cfg, "evaluation")
        

