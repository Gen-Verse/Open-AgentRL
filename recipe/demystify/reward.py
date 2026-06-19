# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
from typing import Any
import ast
import datasets
import json
from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.reward_score.livecodebench import code_math
import numpy as np
logger = logging.getLogger(__name__)


def _to_py(x):
    if isinstance(x, np.generic):      # e.g., np.int64 / np.float64 / np.bool_
        return x.item()
    if isinstance(x, dict):
        return {k: _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    return x

class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics, use Nones
        return result, None, None
answer_format =  "\nRemember once you make sure the current answer is your final answer, do not call the tools again and directly output the final answer in the following text format, the answer format must be: \\boxed{'The final answer goes here.'}."
math_prompt_1 = "Analyze and solve the following math problem step by step. \n\n"
math_prompt_2 = "\n\nThe tool could be used for more precise and efficient calculation and could help you to verify your result before you reach the final answer."
#code_agent_prompt ="Note: You should first analyze the problem carefully and try to use tools to test your code, you can design some simple unit tests to initially verify the correctness of your code. After you make sure that your code is correct, do not call the tool again and directly submit your final code within ```python\n# YOUR CODE HERE\n```"
agent_prompt= "\n\n**Note: You should first analyze the problem and form a high-level solution strategy, then utilize the tools to help you solve the problem.**"

class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)['train']
            if "data_source" in dataframe.column_names:
                data_source = dataframe['data_source'][0]
            if data_source in ["AIME2025", "AIME2024"]:
                dataframe = dataframe.map(
                    self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            elif data_source in ['math_dapo','mega-science']:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            elif data_source == 'gpqa_diamond':
                dataframe = dataframe.map(self.map_fn_gpqa,num_proc=16)
            elif data_source == 'LiveCodeBench_v6':
                dataframe = dataframe.map(self.map_lcb,num_proc=16)
            else:
                dataframe = dataframe.map(self.map_fn_skywork, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "AIME2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source == "AIME2025":
            problem, answer = row["problem"], row["answer"]
        prompt = math_prompt_1 + problem + math_prompt_2 + agent_prompt + answer_format
        data = {
            "data_source": data_source,  # aime_2024, aime_2025
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        return data
    
    def map_fn_gpqa(self, row: dict, *, data_source: str = None):
        problem,answer,domain =row['problem'],row['solution'],row['domain']
        prompt_1 = f"Analyze and solve the following {domain} problem step by step. \n\n"
        prompt = prompt_1 + problem + math_prompt_2 + agent_prompt + answer_format + "\n Here you need to put the final uppercase letter option of this problem into \\boxed{}"
        data = {
            "data_source":row['data_source'],
            "prompt": [{"role": "user", "content": prompt}],
            "ability": domain,
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        return data
    
    def map_lcb(self, row: dict, *, data_source: str = None):
        problem =row['problem']
        start_prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
        code_prompt = "\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```"
        reward_model = json.loads(row['reward_model'])
        inputs = reward_model['ground_truth']['inputs']
        outputs = reward_model['ground_truth']['outputs']
        public_examples = "\nHere are some input and output examples of the expected code:\nInput:" + str(inputs) +"\nOutput:\n" + str(outputs)
        prompt = start_prompt + problem +  public_examples + agent_prompt + code_prompt +"\nBefore sumbit your code, you can utilize tools to check the correctness of your code, once you make sure the current code is correct, do not call the tools again and submit your code within ```python\n# YOUR CODE HERE\n```."
        data = {
            "data_source":row['data_source'],
            "prompt": [{"role": "user", "content": prompt}],
            "ability": row['ability'],
            "reward_model": json.loads(row['reward_model']),
            "agent_name": "tool_agent",
            "extra_info":json.loads(row['extra_info'])
        }
        return data
    
    def map_fn2(self, row: dict):
        content = row["prompt"][0]["content"]
        row["prompt"][0]["content"] = content + "\nDo not put units of the final answer inside \\boxed{}. The content of \\boxed{} should be the numerical value of the final answer only, without any units."
        row["agent_name"] = "tool_agent"
        return row
    
    def map_fn_skywork(self,row:dict):
        content = row["prompt"][0]["content"]
        if "code" in row['data_source']:
            row["prompt"][0]["content"] = content + agent_prompt + "\nBefore sumbit your code, you can utilize tools to check the correctness of your code, once you make sure the current code is correct, do not call the tools again and submit your code within ```python\n# YOUR CODE HERE\n```."
            row["agent_name"] = "tool_agent"
        else:
            row["prompt"][0]["content"] = math_prompt_1 + content + math_prompt_2 + agent_prompt + answer_format + "\nDo not put units of the final answer inside \\boxed{}. The content of \\boxed{} should be the final numerical value of the final answer only, without any units."
            row["agent_name"] = "tool_agent"
        return row


def compute_score(data_source, solution_str, ground_truth, extra_info):
    # use \\boxed{...} answer
    ds = (data_source or "").lower()
    if 'code' in ds:
        result = code_math.compute_score(solution_str, ground_truth)
    else:
        result = math_dapo.compute_score(solution_str=solution_str,ground_truth=ground_truth,strict_box_verify=True)
    num_turns = int(extra_info.get("num_turns",0))
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = float(min(-0.6, result["score"] + tool_call_reward))
    if result["pred"] is None:
        result["pred"] = ""
    return result

def compute_score_outcome_reward(data_source, solution_str, ground_truth, extra_info):
    # use \\boxed{...} answer
    ds = (data_source or "").lower()
    if 'code' in ds:
        result = code_math.compute_score(solution_str, ground_truth)
    else:
        result = math_dapo.compute_score(solution_str=solution_str,ground_truth=ground_truth,strict_box_verify=True)
    num_turns = int(extra_info.get("num_turns",0))
    if result["score"] < 0:
        tool_call_reward = 0
        result["score"] = float(min(-0.6, result["score"] + tool_call_reward))
    if result["pred"] is None:
        result["pred"] = ""
    return result
