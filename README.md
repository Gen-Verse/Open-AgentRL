<div align="center">
<img src="figs/image2.png" width="330">


</div>



### RLAnything & DemyAgent: Open-Source RL for LLMs and Agentic Scenarios









<details>
  <summary>
    <b>RLAnything</b>
    <a href="https://arxiv.org/abs/2602.02488">
      <img src="https://img.shields.io/badge/Paper-Arxiv%202602.02488-red?logo=arxiv&logoColor=red" alt="Paper" height="18" />
    </a>
    <a href="https://huggingface.co/collections/Gen-Verse/open-agentrl">
      <img src="https://img.shields.io/badge/Models-Policy%20&%20Reward-FFCC00?logo=huggingface&logoColor=yellow" alt="Model" height="18" />
    </a>
    <a href="https://yinjjiew.github.io/projects/rlanything/">
      <img src="https://img.shields.io/badge/Blog-RLAnything-blue?logo=rss&logoColor=white" alt="Blog" height="18" />
    </a>
    <b>(click to expand)</b>
  </summary>


<div align="center">
  <h3>
    RLAnything: Forge Environment, Policy, and Reward Model<br>
    in Completely Dynamic RL System
  </h3>
</div>



<table class="center">     <tr>     <td width=100% style="border: none"><img src="figs/rlanythingoverview.png" style="width:100%"></td>     </tr>     <tr>     <td width="100%" style="border: none; text-align: center; word-wrap: break-word">An overview of our research on RLAnything. </td>   </tr> </table>

In this work, we propose RLAnything, a reinforcement learning framework that dynamically optimizes each component through closed-loop optimization, amplifying learning signals and strengthening the overall system:
* The policy is trained with integrated feedback from outcome and step-wise signals from reward model, better than using outcome only.
* Reward model is jointly optimized via consistency feedback, which in turn further improves policy training.
* Our theory-motivated automatic environment adaptation improves training for both the reward and policy models by leveraging critic feedback from each, enabling learning from experience.
* Through extensive experiments, we demonstrate each added component consistently improves the overall system.
* We show that step-wise signals from optimized reward-model outperform outcome signals that rely on human labels.

<p align="center">
  <img src="figs/rlanythingpaperoverview.png" alt="Figure 1" width="600">
</p>

</details>











<details>
  <summary>
    <b>DemyAgent</b>
    <a href="https://arxiv.org/abs/2510.11701">
      <img
        src="https://img.shields.io/badge/Paper-Arxiv%202510.11701-red?logo=arxiv&logoColor=red"
        alt="Paper"
        height="18"
        style="vertical-align: middle;"
      />
    </a>
    <a href="https://huggingface.co/collections/Gen-Verse/open-agentrl-68eda4c05755ca5a8c663656">
      <img
        src="https://img.shields.io/badge/Datasets-Agent%20RL%20Datasets-orange?logo=huggingface&logoColor=yellow"
        alt="Data"
        height="18"
        style="vertical-align: middle;"
      />
    </a>
    <a href="https://huggingface.co/Gen-Verse/DemyAgent-4B">
      <img
        src="https://img.shields.io/badge/DemyAgent%204B-DemyAgent%204B%20Model-FFCC00?logo=huggingface&logoColor=yellow"
        alt="Model"
        height="18"
        style="vertical-align: middle;"
      />
    </a>
    <b>(click to expand)</b>
  </summary>

<div>
<h3>Demystifying Reinforcement Learning in Agentic Reasoning</h3></div>




    
<table class="center">     <tr>     <td width=100% style="border: none"><img src="figs/overview.png" style="width:100%"></td>     </tr>     <tr>     <td width="100%" style="border: none; text-align: center; word-wrap: break-word">An overview of our research on agentic RL. </td>   </tr> </table>

In this work, we systematically investigate three dimensions of agentic RL: **data, algorithms, and reasoning modes**. Our findings reveal: 
* Real end-to-end trajectories and high-diversity datasets significantly outperform synthetic alternatives; 
* Exploration-friendly techniques like reward clipping and entropy maintenance boost training efficiency; 
* Deliberative reasoning with selective tool calls surpasses frequent invocation or verbose self-reasoning. 

We also contribute [high-quality SFT and RL datasets](https://huggingface.co/collections/Gen-Verse/open-agentrl-68eda4c05755ca5a8c663656), demonstrating that **simple recipes enable even [4B models](https://huggingface.co/Gen-Verse/DemyAgent-4B) to outperform 32B models** on challenging benchmarks including AIME2024/2025, GPQA-Diamond, and LiveCodeBench-v6.

</details>

| | **RLAnything** | **DemyAgent** |
|---|---|---|
| **Focus** | Closed-loop RL optimization | Agentic reasoning |
| **Core Idea** | Joint optimization of policy, reward model & environment | Real trajectories + exploration-friendly techniques + deliberative reasoning |
| **Release** | LLM/GUI/Coding Policy & Reward Model | 3K SFT + 30K RL Data, SOTA-level DemyAgent-4B |


## 🚩 New Updates

- **[2026.2]** We fully open-source our work **RLAnything**, including:
  - Training code across GUI Agent, LLM Agent, and Coding LLM settings.
  - Model checkpoints: both the policy models ([RLAnything-7B/8B](https://huggingface.co/collections/Gen-Verse/open-agentrl)) and reward models ([RLAnything-Reward-8B/14B](https://huggingface.co/collections/Gen-Verse/open-agentrl)) across these settings.
  - Evaluation Scripts for our models 


- **[2025.10]** We fully open-source our work **DemyAgent**, including:
  - Training code for both SFT and RL stages
  - High-quality SFT dataset (3K samples) and RL dataset (30K samples)
  - Model checkpoints: SFT models (Qwen2.5-7B-RA-SFT, Qwen3-4B-RA-SFT) and RL-trained model ([DemyAgent-4B](https://huggingface.co/Gen-Verse/DemyAgent-4B))
  - Evaluation Scripts for our models




## 🧭 Navigation


    
- **DemyAgent**:
  - [Get Started](#demyagent-get-started)
  - [Training](#demyagent-train)
    - [Cold-Start SFT](#demyagent-cold-sft) 
    - [Agentic RL](#"demyagent-agent-rl") 
  - [Evaluation](#demyagent-eval)
  - [Results](#demyagent-result)

- **RLAnything**:
  - [Get Started](#rlanything-get-started)
  - [Training](#rlanything-train)
    - [Computer Control](#rlanything-computer-control) 
    - [Text-based Game](#rlanything-text-game) 
    - [RLVR Coding](#rlanything-coding) 
  - [Evaluation](#rlanything-eval)
  - [Results](#rlanything-result)

## 🚀 Get Started

<a id="demyagent-get-started"></a>
### DemyAgent

```bash
git clone https://github.com/Gen-Verse/Open-AgentRL.git
conda create -n OpenAgentRL python=3.11 
conda activate OpenAgentRL
cd Open-AgentRL
bash scripts/install_vllm_sglang_mcore.sh
pip install -e .[vllm]
```

<a id="rlanything-get-started"></a>
### RLAnything

```bash
conda create --name rlanything python=3.10
source activate rlanything
pip install -r requirements_rlanything.txt
```

<a id="demyagent-train"></a>
## 🔧 DemyAgent Training



<a id="demyagent-cold-sft"></a>
### Cold-Start SFT

Before you start SFT, make sure you have downloaded the [3K Agentic SFT Data](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-SFT-3K) and the corresponding base models like [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). Configure [qwen3_4b_sft.sh](recipe/demystify/qwen3_4b_sft.sh) and [qwen2_7b_sft.sh](recipe/demystify/qwen2_7b_sft.sh), and set the absolute paths to your model and the `.parquet` data files.

- [🤗 3K Agentic SFT Data](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-SFT-3K)

- **TRAIN_DATA**: Path to the `.parquet` file of the SFT dataset
- **EVAL_DATA**: Path to the evaluation data (can be set to the same as **TRAIN_DATA**)
- **MODEL_PATH**: Path to your base models like [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) or [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **SAVE_PATH**: Directory to save the SFT model checkpoints

After all configurations are set, simply run the code below to finetune Qwen3-4B-Instruct-2507:

```bash
bash recipe/demystify/qwen3_4b_sft.sh
```

After obtaining the SFT models, utilize the following command to merge the model:

```bash
python3 -m verl.model_merger merge --backend fsdp --local_dir xxx/global_step_465 --target_dir xxx/global_step_465/huggingface
```
<a id="demyagent-agent-rl"></a>
### Agentic RL

- [🤗 30K Agentic RL Data](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K)

After obtaining the SFT models (you can also directly use our provided checkpoints [Qwen2.5-7B-RA-SFT](https://huggingface.co/Gen-Verse/Qwen2.5-7B-RA-SFT) and [Qwen3-4B-RA-SFT](https://huggingface.co/Gen-Verse/Qwen3-4B-RA-SFT)), you can start Agentic RL with our GRPO-TCR recipe.

First, download our [30K Agentic RL Data](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K) and the [evaluation datasets](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-Eval).

Then, configure the [SandboxFusion](https://github.com/bytedance/SandboxFusion) environment for code execution.

There are two ways to create a sandbox:

1. **Local Deployment**: Deploy SandboxFusion locally by referring to [the SandboxFusion deployment documentation](https://bytedance.github.io/SandboxFusion/docs/docs/get-started#local-deployment)
2. **Cloud Service**: Use Volcano Engine Cloud FaaS service by referring to [Volcano Engine Code Sandbox](https://www.volcengine.com/docs/6662/1539235)

Using either method, obtain an API endpoint (something like `https://<ip-address-or-domain-name>/run_code`), and configure it in **`recipe/demystify/sandbox_fusion_tool_config.yaml`** and **the function check_correctness in`verl/utils/reward_score/livecodebench/code_math.py`**.

Next, configure the Agentic RL scripts [grpo_tcr_qwen2_7b.sh](recipe/demystify/grpo_tcr_qwen2_7b.sh) and [grpo_tcr_qwen3_4b.sh](recipe/demystify/grpo_tcr_qwen3_4b.sh):

- **open_agent_rl**: Path to the `.parquet` file of the agentic RL dataset
- **model_path**: Path to the SFT models
- **aime2024/aime2025**: Benchmark datasets evaluated every 10 training steps. Set the absolute paths to the `.parquet` files of the benchmarks. You can also add more benchmarks like GPQA-Diamond in **test_files**
- **default_local_dir**: Directory to save your RL checkpoints

**Training Resources**: We conducted our training on one $8\times \text{Tesla-A100}$ node with a batch size of 64.

After finishing the configurations, run the code below to conduct Agentic RL with the GRPO-TCR recipe:

```bash
bash recipe/demystify/grpo_tcr_qwen3_4b.sh
```

You can observe the training dynamics and evaluation results in Weights & Biases (wandb).



<a id="rlanything-train"></a>
## 🔧 RLAnything Training


<a id="rlanything-computer-control"></a>
### Computer Control (OSWorld)

Our reinforcement learning and evaluation pipeline for OSWorld is built on a pool of virtual machines running in parallel on cloud instances. Specifically, we use Volcengine Cloud in our experiments. Before training, you need to set up the security group and VM image on Volcengine following [these instructions](https://github.com/xlang-ai/OSWorld/blob/main/desktop_env/providers/volcengine/VOLCENGINE_GUIDELINE_CN.md).

Before training, you need set `osworld_rl.yaml` in configs. The detailed instructions are within it. To start the RLAnything training, simply
```bash
python osworld_rl.py config=configs/osworld_rl.yaml
# you need to set num_node in osworld_rl.yaml to 1 if you only use one node.
```
In our experiments, we train with multiple nodes:
```bash
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python osworld_rl.py config=configs/osworld_rl.yaml
else
    exec tail -f /dev/null
fi
# directly submit this to head machine
```



<a id="rlanything-text-game"></a>
### Text-based Game (AlfWorld)

To conduct reinforcement learning or evaluation on AlfWorld, you need to first download the AlfWorld data with the following commands (after you have pip installed the rlanything environment)
```bash
alfworld-download
```
Then you will have a directory that contains at least `detectors`, `json_2.1.1`, and `logic`. This will be the directory to save AlfWorld environment files. Our adapted environment files will be generated under `alfworld_file_path/json_2.1.1/alfworld_rl/syn_train`. `syn_train` saves accepted environment files, while `temp_train` saves generated files which to be validated (not accept yet).

Before training, you need set `alfworld_rl.yaml` in configs. The detailed instructions are within it. To start the RLAnything training, simply
```bash
python alfworld_rl.py config=configs/alfworld_rl.yaml
# you need to set num_node in alfworld_rl.yaml to 1 if you only use one node.
```
In our experiments, we train with multiple nodes:
```bash
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python alfworld_rl.py config=configs/alfworld_rl.yaml
else
    exec tail -f /dev/null
fi
# directly submit this to head machine
```



<a id="rlanything-coding"></a>
### Coding

You need to first download the training and evaluation dataset. Simply open `./data` and follow the instructions to do so. Then you can start the training with
```bash
python coding_rl.py config=configs/coding_rl.yaml
# you need to set num_node in coding_rl.yaml to 1 if you only use one node.
```
In our experiments, we train on 4 nodes
```bash
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python coding_rl.py config=configs/coding_rl.yaml
else
    exec tail -f /dev/null
fi
```






<a id="demyagent-eval"></a>
## 📊 DemyAgent Evaluation



If you have already trained a model, you can refer to the following process for agentic reasoning capability evaluation. Alternatively, you can download our checkpoint from [🤗 DemyAgent-4B](https://huggingface.co/Gen-Verse/DemyAgent-4B) for direct testing.

### AIME2024/2025 and GPQA-Diamond

Configure the scripts [eval_qwen2_7b_aime_gpqa.sh](recipe/demystify/eval/eval_qwen2_7b_aime_gpqa.sh) and [eval_qwen3_4b_aime_gpqa.sh](recipe/demystify/eval/eval_qwen3_4b_aime_gpqa.sh). The configuration process is similar to the training setup—set the paths to your models and `.parquet` files of the benchmarks.

Simply run the code below to evaluate performance on AIME2024/2025 and GPQA-Diamond:

```bash
bash recipe/demystify/eval/eval_qwen3_4b_aime_gpqa.sh
```

You can observe the average@32/pass@32/maj@32 metrics from your wandb project.

### LiveCodeBench-v6

First, run inference for the benchmark:

```bash
bash recipe/demystify/eval/eval_qwen3_4b_livecodebench.sh
```

Specifically, we save the validation rollout paths in **VAL_SAVE_PATH**. After obtaining the validation rollouts, refer to the official evaluation process for local results in [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).


<a id="rlanything-eval"></a>
## 📊 RLAnything Evaluation

### Computer Control (OSWorld)

To eval the model on OSWorld, use
```bash
python osworld_eval.py config=configs/osworld_eval.yaml
```
You can also evaluate with multi-nodes to accelerate.
```bash
if [[ ${MLP_ROLE_INDEX:-0} -eq 0 ]]; then   
    python osworld_eval.py config=configs/osworld_eval.yaml
else
    exec tail -f /dev/null
fi
```



### Text-based Game (AlfWorld)

To eval the model on AlfWorld, use
```bash
python alfworld_eval.py config=configs/alfworld_eval.yaml
```

### Coding

For evaluation, simply
```bash
python coding_eval.py config=configs/coding_eval.yaml
```



<a id="demyagent-result"></a>
## 📈 DemyAgent Results


We provide the evaluation results of the agentic reasoning abilities of our models on challenging benchmarks including AIME2024/AIME2025, GPQA-Diamond, and LiveCodeBench-v6.

|                            | **MATH**     |              | **Science**      | **Code**             |
| -------------------------- | ------------ | ------------ | ---------------- | -------------------- |
| **Method**                 | **AIME2024** | **AIME2025** | **GPQA-Diamond** | **LiveCodeBench-v6** |
| *Self-Contained Reasoning* |              |              |                  |                      |
| Qwen2.5-7B-Instruct        | 16.7         | 10.0         | 31.3             | 15.2                 |
| Qwen3-4B-Instruct-2507     | 63.3         | 47.4         | 52.0             | **35.1**             |
| Qwen2.5-72B-Instruct       | 18.9         | 15.0         | 49.0             | -                    |
| DeepSeek-V3                | 39.2         | 28.8         | 59.1             | 16.1                 |
| DeepSeek-R1-Distill-32B    | 70.0         | 46.7         | 59.6             | -                    |
| DeepSeek-R1-Zero (671B)    | 71.0         | 53.5         | 59.6             | -                    |
| *Agentic Reasoning*        |              |              |                  |                      |
| Qwen2.5-7B-Instruct        | 4.8          | 5.6          | 25.5             | 12.2                 |
| Qwen3-4B-Instruct-2507     | 17.9         | 16.3         | 44.3             | 23.0                 |
| ToRL-7B                    | 43.3         | 30.0         | -                | -                    |
| ReTool-32B                 | 72.5         | 54.3         | -                | -                    |
| Tool-Star-3B               | 20.0         | 16.7         | -                | -                    |
| ARPO-7B                    | 30.0         | 30.0         | 53.0             | 18.3                 |
| rStar2-Agent-14B           | **80.6**     | <u>69.8</u>  | **60.9**         | -                    |
| **DemyAgent-4B (Ours)**    | <u>72.6</u>  | **70.0**     | <u>58.5</u>      | <u>26.8</u>          |

As demonstrated in the table above, despite having only 4B parameters, **DemyAgent-4B** matches or even outperforms much larger models (14B/32B) across challenging benchmarks. Notably, **DemyAgent-4B achieves state-of-the-art agentic reasoning performance**, surpassing [ReTool-32B](https://arxiv.org/pdf/2504.11536) and [rStar2-Agent-14B](https://arxiv.org/pdf/2508.20722), and even outperforming long-CoT models like DeepSeek-R1-Zero on AIME2025, which further validates the insights of our research.


<a id="rlanything-result"></a>
## 📈 RLAnything Results

In the following table, we demonstrate the effectiveness of the **RLAnything** framework. Each added dynamic component contributes to improvements in the overall system.

<p align="center">
  <img src="figs/rlanythingmaintable.png" alt="Figure 1" width="800">
</p>

We further scale the optimization for GUI agent and achieves SoTA results:

<p align="center">
  <img src="figs/rlanythingscaleosworld.png" alt="Figure 1" width="600">
</p>

<p align="center">
  <img src="figs/rlanythingosworldbench.png" alt="Figure 1" width="800">
</p>


## 📝 Citation

```bibtex
@article{yu2025demystify,
  title={Demystifying Reinforcement Learning in Agentic Reasoning},
  author={Yu, Zhaochen and Yang, Ling and Zou, Jiaru and Yan, Shuicheng and Wang, Mengdi},
  journal={arXiv preprint arXiv:2510.11701},
  year={2025}
}
```

## 🙏 Acknowledgements

This work aims to explore more efficient paradigms for Agentic RL. Our implementation builds upon the excellent codebases of [VeRL](https://github.com/volcengine/verl) and [ReTool](https://github.com/ReTool-RL/ReTool). We sincerely thank these projects for their valuable insights and high-quality implementations, which have greatly facilitated our research.



















































































