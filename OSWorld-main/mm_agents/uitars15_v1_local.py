# -*- coding: utf-8 -*-
"""
uitars15_v1_local.py

把原始 uitars15_v1.py 改成 “local-only + 可序列化 trajectory + reward_trajectory + record_step_outcome + summarize”
并且补齐你指出缺失的：
- reward_message（每步都会生成、存储、并返回在 other_info 里）
- record_step_outcome(next_obs)
- summarize(result)

对齐你给的 qwen3vl_agent_local.py 设计：
- 本地 OpenAI-compatible 端点（httpx，支持多端点、pid sticky、不允许 fallback）
- trajectory / reward_trajectory 存为可 JSON 序列化的 messages（image_url/image -> step_k.png）
- reward prompt 复用 mm_agents.opencua.prompts 里的版本
"""

from __future__ import annotations

import ast
import base64
import copy
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from loguru import logger
from PIL import Image

from mm_agents.accessibility_tree_wrap.heuristic_retrieve import filter_nodes

# reward prompts：与你的 qwen3vl_agent_local.py 对齐（直接复用 OpenCUA 的 prompts）
from mm_agents.opencua.prompts import (
    REWARD_SYSTEM_PROMPT_V1_L2,
    REWARD_INSTRUTION_TEMPLATE,
    REWARD_INSTRUTION_TEMPLATE_POST,
)

# =========================
# UI-TARS action spaces / prompts (保持原样)
# =========================

UITARS_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
"""

UITARS_CALL_USR_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
"""

UITARS_NORMAL_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.
"""

UITARS_USR_PROMPT_NOTHOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Action: ...
```
## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
## User Instruction
{instruction}
"""

UITARS_USR_PROMPT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


# =========================
# Parsing helpers (保持原样)
# =========================

def parse_action(action_str):
    try:
        node = ast.parse(action_str, mode="eval")
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        call = node.body
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {"function": func_name, "args": kwargs}

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None


def escape_single_quotes(text: str) -> str:
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def linear_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if width * height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(height * resize_factor)
    return height, width


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_action_to_structure_output(
    text,
    factor,
    origin_resized_height,
    origin_resized_width,
    model_type,
    max_pixels=16384 * 28 * 28,
    min_pixels=100 * 28 * 28,
):
    text = (text or "").strip()
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height,
            origin_resized_width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"

    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()

    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:

            def escape_quotes(match):
                content = match.group(1)
                return content

            pattern = r"type\(content='(.*?)'\)"
            content = re.sub(pattern, escape_quotes, action_str)

            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n", "\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance is None:
            print(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}")

        action_type = action_instance["function"]
        params = action_instance["args"]

        action_inputs = {}
        for param_name, param in params.items():
            if param == "":
                continue
            param = param.lstrip()
            action_inputs[param_name.strip()] = param

            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num / smart_resize_height))
                        else:
                            float_numbers.append(float(num / smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]

                action_inputs[param_name.strip()] = str(float_numbers)

        actions.append(
            {
                "reflection": reflection,
                "thought": thought,
                "action_type": action_type,
                "action_inputs": action_inputs,
                "text": text,
            }
        )
    return actions


def parsing_response_to_pyautogui_code(responses, image_height: int, image_width: int, input_swap: bool = True) -> str:
    pyautogui_code = "import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        observation = response.get("observation", "")
        thought = response.get("thought", "")

        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += "\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})

        if action_type == "hotkey":
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"
            elif hotkey == "arrowright":
                hotkey = "right"
            elif hotkey == "arrowup":
                hotkey = "up"
            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                keys = hotkey.split()
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = " "
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"

        elif action_type == "press":
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            # NOTE: 原文件里这里用 hotkey 变量（未定义）是个潜在 bug；为了不引入行为变化，这里不改动逻辑结构。
            if key_to_press:
                pyautogui_code += f"\npyautogui.press({repr(key_to_press)})"

        elif action_type == "keyup":
            key_to_up = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyUp({repr(key_to_up)})"

        elif action_type == "keydown":
            key_to_down = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyDown({repr(key_to_down)})"

        elif action_type == "type":
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += "\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += "\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += "\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += "\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += "\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += "\npyautogui.press('enter')"

        elif action_type in ["drag", "select"]:
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += f"\npyautogui.moveTo({sx}, {sy})\n"
                pyautogui_code += f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"

        elif action_type == "scroll":
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")

            if x is None:
                if "up" in direction.lower():
                    pyautogui_code += "\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += "\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"

        elif action_type in ["finished"]:
            pyautogui_code = "DONE"

        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code


def add_box_token(input_string: str) -> str:
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            updated_action = action
            for coord_type, x, y in coordinates:
                updated_action = updated_action.replace(
                    f"{coord_type}='({x},{y})'",
                    f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'",
                )
            processed_actions.append(updated_action)
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):
    # 注意：原文件里 attributes_ns_ubuntu 等常量未必存在；
    # 且 predict 里对该函数是 try/except，所以这里保持原样，不强行补全常量定义。
    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu  # noqa: F821
        _component_ns = component_ns_ubuntu  # noqa: F821
        _value_ns = value_ns_ubuntu  # noqa: F821
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows  # noqa: F821
        _component_ns = component_ns_windows  # noqa: F821
        _value_ns = value_ns_windows  # noqa: F821
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = [
        "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"
    ]

    for node in filtered_nodes:
        if node.text:
            text = node.text if '"' not in node.text else '"{:}"'.format(node.text.replace('"', '""'))
        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith("EditWrapper") and node.get(  # noqa: F821
            "{{{:}}}value".format(_value_ns)
        ):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = node_text if '"' not in node_text else '"{:}"'.format(node_text.replace('"', '""'))
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag,
                node.get("name", ""),
                text,
                (
                    node.get("{{{:}}}class".format(_attributes_ns), "")
                    if platform == "ubuntu"
                    else node.get("{{{:}}}class".format(class_ns_windows), "")  # noqa: F821
                ),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get("{{{:}}}screencoord".format(_component_ns), ""),
                node.get("{{{:}}}size".format(_component_ns), ""),
            )
        )

    return "\n".join(linearized_accessibility_tree)


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    return linearized_accessibility_tree


# =========================
# Local-only helpers (新增，对齐 qwen3vl_agent_local.py)
# =========================

def convert_message_with_img(messages: List[Dict[str, Any]], images: List[str]) -> List[Dict[str, Any]]:
    """
    把 messages 里的 image_url / image 替换为可序列化的 image path（step_x.png）。
    与 qwen3vl_agent_local.py 的 convert_message_with_img 同语义。
    """
    msgs_no_img: List[Dict[str, Any]] = []
    images_i = 0
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")

        if isinstance(content, list):
            new_content = []
            for part in content:
                t = part.get("type")
                if t in ("image_url", "image"):
                    if images_i >= len(images):
                        raise RuntimeError(
                            f"convert_message_with_img: need >= {images_i+1} images, but only {len(images)}. "
                            f"(role={role}, images_i={images_i})"
                        )
                    img_val = images[images_i]
                    new_content.append({"type": "image", "image": img_val})
                    images_i += 1
                elif t == "text":
                    new_content.append(part)
                else:
                    # 容忍未知 part
                    new_content.append(part)
            msgs_no_img.append({"role": role, "content": new_content})
        else:
            msgs_no_img.append(m)

    return msgs_no_img


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_pil_png_if_missing(pil_img: Image.Image, path: str) -> None:
    """
    为了保证 trajectory.json 里引用的 step_k.png 一定存在：
    - 若文件已存在，不覆盖（避免外部环境保存原始图时被我们覆盖）。
    - 若不存在，则用“模型实际看到的 resized 图”写入。
    """
    try:
        if os.path.exists(path):
            return
        _ensure_dir(os.path.dirname(path))
        pil_img.save(path, format="PNG")
    except Exception as _e:
        # 不影响主流程
        pass


def _save_bytes_png_if_missing(image_bytes: bytes, path: str) -> None:
    try:
        if os.path.exists(path):
            return
        _ensure_dir(os.path.dirname(path))
        img = Image.open(BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(path, format="PNG")
    except Exception:
        pass


# =========================
# Main agent (local)
# =========================

class UITARSAgentLocal:
    def __init__(
        self,
        model: str,
        runtime_conf: Dict,
        platform: str = "ubuntu",
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        max_trajectory_length: int = 50,
        a11y_tree_max_tokens: int = 10000,
        model_type: str = "qwen25vl",
        example_result_dir: Optional[str] = None,
        max_steps: Optional[int] = None,
        max_image_history_length: Optional[int] = None,
        max_reward_image_history_length: int = 1,
        **kwargs,
    ):
        self.model = model
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.model_type = model_type

        self.runtime_conf = runtime_conf
        self.temperature = self.runtime_conf["temperature"]
        self.top_k = self.runtime_conf.get("top_k", -1)
        self.top_p = self.runtime_conf["top_p"]
        self.max_tokens = self.runtime_conf["max_tokens"]
        self.infer_mode = self.runtime_conf["infer_mode"]
        self.prompt_style = self.runtime_conf["prompt_style"]
        self.input_swap = self.runtime_conf["input_swap"]
        self.language = self.runtime_conf["language"]
        self.max_pixels = self.runtime_conf["max_pixels"]
        self.min_pixels = self.runtime_conf["min_pixels"]
        self.callusr_tolerance = self.runtime_conf["callusr_tolerance"]

        self.history_n = self.runtime_conf.get("history_n", 5)
        self.max_steps = max_steps if max_steps is not None else self.runtime_conf.get("max_steps", max_trajectory_length)
        self.max_image_history_length = (
            max_image_history_length if max_image_history_length is not None else self.runtime_conf.get("max_image_history_length", self.history_n)
        )
        self.max_reward_image_history_length = self.runtime_conf.get("max_reward_image_history_length", max_reward_image_history_length)

        self.example_result_dir = example_result_dir or self.runtime_conf.get("example_result_dir") or os.getcwd()

        # history
        self.thoughts: List[str] = []
        self.actions: List[List[str]] = []          # 每步返回的 pyautogui code list
        self.observations: List[Dict[str, Any]] = []
        self.history_images: List[bytes] = []       # raw screenshot bytes
        self.history_responses: List[str] = []      # raw model output (Thought/Action string)

        # prompts
        self.prompt_action_space = UITARS_ACTION_SPACE
        self.action_parse_res_factor = 1000
        if self.infer_mode == "qwen2vl_user":
            self.prompt_action_space = UITARS_CALL_USR_ACTION_SPACE
        elif self.infer_mode == "qwen25vl_normal":
            self.prompt_action_space = UITARS_NORMAL_ACTION_SPACE

        self.prompt_template = UITARS_USR_PROMPT_THOUGHT
        if self.prompt_style in ("qwen2vl_user", "qwen25vl_normal"):
            self.prompt_template = UITARS_USR_PROMPT_THOUGHT
        elif self.prompt_style == "qwen2vl_no_thought":
            self.prompt_template = UITARS_USR_PROMPT_NOTHOUGHT

        self.cur_callusr_count = 0

        # --- trajectory saving (对齐 qwen3vl_agent_local.py / OpenCUAAgentLocal) ---
        self.trajectory: List[Dict[str, Any]] = []
        self.reward_trajectory: List[Dict[str, Any]] = []

    def reset(self, runtime_logger=None):
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.trajectory = []
        self.reward_trajectory = []
        self.cur_callusr_count = 0

    # -------------------------
    # local LLM call (no fallback)
    # -------------------------
    def call_llm(self, payload: Dict[str, Any], model: str) -> str:
        """
        local-only + 多端点 + pid sticky + 不 fallback（对齐 qwen3vl_agent_local.py）

        Env:
        - UITARS_LOCAL_ENDPOINTS / UITARS15_LOCAL_ENDPOINTS / OPENCUA_LOCAL_ENDPOINTS: 逗号分隔多个端点
        - UITARS_LOCAL_ENDPOINT  / UITARS15_LOCAL_ENDPOINT  / OPENCUA_LOCAL_ENDPOINT : 单个端点
        - API Key：UITARS_API_KEY / UITARS15_API_KEY / OPENCUA_API_KEY（都没有就 dummy）

        端点需 OpenAI-compatible: POST {base}/v1/chat/completions
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('UITARS_API_KEY') or os.environ.get('UITARS15_API_KEY') or os.environ.get('OPENCUA_API_KEY', 'dummy')}",
        }

        local_eps = (
            os.getenv("UITARS_LOCAL_ENDPOINTS")
            or os.getenv("UITARS15_LOCAL_ENDPOINTS")
            or os.getenv("OPENCUA_LOCAL_ENDPOINTS")
        )
        local_ep = (
            os.getenv("UITARS_LOCAL_ENDPOINT")
            or os.getenv("UITARS15_LOCAL_ENDPOINT")
            or os.getenv("OPENCUA_LOCAL_ENDPOINT")
        )

        bases: List[str] = []
        if local_eps:
            eps = [e.strip() for e in local_eps.split(",") if e.strip()]
            if eps:
                idx = os.getpid() % len(eps)
                bases = [eps[idx]] + eps[idx + 1 :] + eps[:idx]
        elif local_ep:
            bases = [local_ep.strip()]

        if not bases:
            looks_like_path = ("/" in str(self.model)) or str(self.model).startswith((".", "/"))
            if looks_like_path:
                raise RuntimeError(
                    "UITARS_LOCAL_ENDPOINTS/UITARS_LOCAL_ENDPOINT (or UITARS15_/OPENCUA_) not set, "
                    "and model looks like a local path. Refusing to fallback to any paid/remote endpoint."
                )
            raise RuntimeError("No local endpoint configured for UITARSAgentLocal.")

        max_total_attempts = 20
        backoff_sec = 5

        for _ in range(max_total_attempts):
            for base in bases:
                url = f"{base.rstrip('/')}/v1/chat/completions"
                try:
                    resp = httpx.post(url, headers=headers, json=payload, timeout=3600, verify=False)
                except Exception as ex:
                    logger.error(f"HTTP error calling {url}: {ex}")
                    continue

                if resp.status_code != 200:
                    logger.error(f"LLM HTTP {resp.status_code} from {url}: {resp.text[:500]}")
                    continue

                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = msg.get("content", "")

                # OpenAI-compatible：一般 finish_reason=stop
                if choice.get("finish_reason") in (None, "stop"):
                    return content or ""
                else:
                    logger.error(
                        f"LLM did not finish properly from {url} (finish_reason={choice.get('finish_reason')}); will retry."
                    )

            time.sleep(backoff_sec)

        raise RuntimeError(f"LLM call failed after {max_total_attempts} attempts across endpoints: {bases}")

    # -------------------------
    # main predict
    # -------------------------
    def predict(self, instruction: str, obs: Dict, last_action_after_obs: Dict = None, **kwargs) -> Tuple[str, List[str], Dict]:
        """
        返回 (prediction_text, pyautogui_actions, other_info)
        - prediction_text: 模型原始输出（Thought/Action 文本）
        - pyautogui_actions: ["DONE"] / ["WAIT"] / ["FAIL"] 或一组 pyautogui code strings
        - other_info: 包含 reward_message / tool parsing 中间产物 / trajectory 索引等
        """
        other_info: Dict[str, Any] = {}

        # step index：用 history_responses 的长度做“已完成 step 数”
        step_index = len(self.history_responses)

        # append raw image history
        screenshot_bytes = obs["screenshot"]
        if isinstance(screenshot_bytes, str):
            # 如果上层给的是 base64 string，尽量解码
            try:
                screenshot_bytes = base64.b64decode(screenshot_bytes)
            except Exception:
                raise TypeError("obs['screenshot'] is str but not valid base64")
        if not isinstance(screenshot_bytes, (bytes, bytearray)):
            raise TypeError(f"obs['screenshot'] must be bytes/base64-str, got {type(obs['screenshot'])}")

        self.history_images.append(bytes(screenshot_bytes))

        # record observation (保持原逻辑)
        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            base64_image = bytes(screenshot_bytes)
            try:
                linearized_accessibility_tree = (
                    linearize_accessibility_tree(
                        accessibility_tree=obs["accessibility_tree"],
                        platform=self.platform,
                    )
                    if self.observation_type == "screenshot_a11y_tree"
                    else None
                )
            except Exception:
                linearized_accessibility_tree = None

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, self.a11y_tree_max_tokens
                )

            if self.observation_type == "screenshot_a11y_tree":
                self.observations.append({"screenshot": base64_image, "accessibility_tree": linearized_accessibility_tree})
            else:
                self.observations.append({"screenshot": base64_image, "accessibility_tree": None})
        else:
            raise ValueError("Invalid observation_type type: " + self.observation_type)

        # build user prompt (保持原样)
        if self.infer_mode in ("qwen2vl_user", "qwen25vl_normal"):
            user_prompt = self.prompt_template.format(
                instruction=instruction,
                action_space=self.prompt_action_space,
                language=self.language,
            )
        elif self.infer_mode == "qwen2vl_no_thought":
            user_prompt = self.prompt_template.format(instruction=instruction)
        else:
            user_prompt = self.prompt_template.format(
                instruction=instruction,
                action_space=self.prompt_action_space,
                language=self.language,
            )

        # keep only last history_n images in memory for prompt building (保持原样)
        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        # build PIL images (保持原样，但要同时准备 step_k.png 路径用于 trajectory)
        images: List[Image.Image] = []
        if isinstance(self.history_images, bytes):
            self.history_images = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            self.history_images = list(self.history_images)
        elif isinstance(self.history_images, list):
            pass
        else:
            raise TypeError(f"Unidentified images type: {type(self.history_images)}")

        for turn, img_bytes in enumerate(self.history_images):
            if len(images) >= self.history_n:
                break
            try:
                img = Image.open(BytesIO(img_bytes))
            except Exception as e:
                raise RuntimeError(f"Error opening image: {e}")

            if img.width * img.height > self.max_pixels:
                resize_factor = math.sqrt(self.max_pixels / (img.width * img.height))
                width, height = int(img.width * resize_factor), int(img.height * resize_factor)
                img = img.resize((width, height))
            if img.width * img.height < self.min_pixels:
                resize_factor = math.sqrt(self.min_pixels / (img.width * img.height))
                width, height = math.ceil(img.width * resize_factor), math.ceil(img.height * resize_factor)
                img = img.resize((width, height))

            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        # base messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

        # 关键：为 convert_message_with_img 维护一个“本次真正送入 messages 的 image 路径序列”
        image_traj: List[str] = []

        # 当前 history_images 对应的全局 step id：只知道“最后 N 张”，所以用 step_index 推回去
        # history_images 里最后一张就是当前 step 的 obs（step_index）
        # 如果 images 有 K 张，则它们对应 step_id = step_index-(K-1) ... step_index
        base_step_id = step_index - (len(images) - 1)

        image_num = 0
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # send at most history_n images to the model（保持原逻辑）
                if history_idx + self.history_n > len(self.history_responses):
                    cur_image = images[image_num]
                    encoded_string = pil_to_base64(cur_image)

                    step_id = base_step_id + image_num
                    step_path = os.path.join(self.example_result_dir, f"step_{step_id}.png")
                    _save_pil_png_if_missing(cur_image, step_path)
                    image_traj.append(step_path)

                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}],
                        }
                    )
                    image_num += 1

                messages.append(
                    {"role": "assistant", "content": [{"type": "text", "text": add_box_token(history_response)}]}
                )

            # append current image
            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)

            step_id = base_step_id + image_num
            step_path = os.path.join(self.example_result_dir, f"step_{step_id}.png")
            _save_pil_png_if_missing(cur_image, step_path)
            image_traj.append(step_path)

            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}],
                }
            )
            image_num += 1

        else:
            # only current image
            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)

            step_id = base_step_id + image_num
            step_path = os.path.join(self.example_result_dir, f"step_{step_id}.png")
            _save_pil_png_if_missing(cur_image, step_path)
            image_traj.append(step_path)

            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}],
                }
            )
            image_num += 1

        # retry loop（保持你原来的：3次 + parse失败调 temperature）
        try_times = 3
        origin_resized_height = images[-1].height
        origin_resized_width = images[-1].width

        temperature = self.temperature
        prediction: Optional[str] = None
        parsed_responses: Optional[List[Dict[str, Any]]] = None

        while True:
            if try_times <= 0:
                err = "Reach max retry times to fetch/parse response from client."
                return err, ["FAIL"], {"error": err}

            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "frequency_penalty": 1,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": self.top_p,
                }
                raw = self.call_llm(payload, self.model)
                prediction = (raw or "").strip()

            except Exception as e:
                logger.exception(f"Error when fetching response from client: {e}")
                prediction = None
                try_times -= 1
                continue

            try:
                parsed_responses = parse_action_to_structure_output(
                    prediction,
                    self.action_parse_res_factor,
                    origin_resized_height,
                    origin_resized_width,
                    self.model_type,
                    self.max_pixels,
                    self.min_pixels,
                )
                break
            except Exception as e:
                logger.error(f"Error when parsing response from client: {e}")
                prediction = None
                parsed_responses = None
                try_times -= 1
                temperature = 1  # 提高随机性避免 parse 崩
                self.top_k = -1

        if prediction is None or parsed_responses is None:
            err = "client error"
            return err, ["FAIL"], {"error": err}

        # 记录 history（保持你原始逻辑）
        self.history_responses.append(prediction)
        self.thoughts.append(prediction)

        # ============ reward prompt（新增：对齐 qwen3vl_agent_local.py） ============
        reward_user_content: List[Dict[str, Any]] = []
        reward_image_traj: List[str] = []

        rstart_i = max(0, step_index - self.max_reward_image_history_length + 1)

        # 这里用 “历史模型输出” 作为 action 描述（比 pyautogui code 更像你要的 reward supervision）
        prev_lines = []
        for i in range(rstart_i):
            prev_lines.append(f"Step {i+1}: {self.history_responses[i]}")
        previous_reward_actions_str = "\n".join(prev_lines) if prev_lines else "None"
        reward_user_content.append({"type": "text", "text": f"Previous Actions:\n {previous_reward_actions_str}"})

        for i in range(rstart_i, step_index):
            reward_user_content.append({"type": "text", "text": "Image of environment:\n"})
            reward_user_content.append({"type": "image", "image": "<image>"})
            reward_image_traj.append(os.path.join(self.example_result_dir, f"step_{i}.png"))
            reward_user_content.append({"type": "text", "text": f"\nAction of agent:\nStep {i+1}:\n{self.history_responses[i]}\n"})

        # current obs
        reward_user_content.append({"type": "text", "text": "Agent's current observation:\n"})
        reward_user_content.append({"type": "image", "image": "<image>"})
        reward_image_traj.append(os.path.join(self.example_result_dir, f"step_{step_index}.png"))

        # reward_message（你点名要出现的变量）
        reward_message = REWARD_INSTRUTION_TEMPLATE.format(instruction=instruction, response=prediction)
        other_info["reward_message"] = reward_message

        reward_user_content.append({"type": "text", "text": "\n" + reward_message})

        reward_messages: List[Dict[str, Any]] = [{"role": "system", "content": REWARD_SYSTEM_PROMPT_V1_L2}]
        reward_messages.append({"role": "user", "content": reward_user_content})

        # ============ parse -> pyautogui actions（保持原样） ============
        actions: List[str] = []
        last_image = Image.open(BytesIO(self.history_images[-1]))
        obs_image_height = last_image.height
        obs_image_width = last_image.width

        # 终止码（保持原逻辑：如果出现 finished/wait/error_env/call_user，直接返回单一控制 token）
        terminal: Optional[str] = None

        for parsed_response in parsed_responses:
            if "action_type" in parsed_response:
                atype = parsed_response["action_type"]

                if atype == FINISH_WORD:
                    terminal = "DONE"
                    break

                if atype == WAIT_WORD:
                    terminal = "WAIT"
                    break

                if atype == ENV_FAIL_WORD:
                    terminal = "FAIL"
                    break

                if atype == CALL_USER:
                    if self.callusr_tolerance > self.cur_callusr_count:
                        self.cur_callusr_count += 1
                        terminal = "WAIT"
                    else:
                        terminal = "FAIL"
                    break

            pyautogui_code = parsing_response_to_pyautogui_code(
                parsed_response,
                obs_image_height,
                obs_image_width,
                self.input_swap,
            )
            actions.append(pyautogui_code)

        # 本步 trajectory 记录（必须在 return 前完成）
        # messages / reward_messages 都转成可序列化
        try:
            self.trajectory.append(
                {
                    "messages": convert_message_with_img(messages, image_traj),
                    "response": prediction,
                    "step_index": step_index,
                    "original_screen_size": [obs_image_width, obs_image_height],
                    "resized_screen_size": [origin_resized_width, origin_resized_height],
                    "parsed_responses": parsed_responses,
                    "pyautogui_actions": actions if actions else ([terminal] if terminal else []),
                }
            )
            self.reward_trajectory.append(
                {
                    "reward_messages": convert_message_with_img(reward_messages, reward_image_traj),
                    "reward_message": reward_message,
                    "step_index": step_index,
                }
            )
        except Exception as e:
            # 不影响主流程，但要把错误带回去方便你 debug
            other_info["trajectory_error"] = str(e)

        # 写入 action history（保持原文件：每步 append actions list / 即使 terminal 也 append 空 actions list）
        self.actions.append(actions)

        # max steps（对齐你 qwen3vl_agent_local.py：强制 FAIL）
        if len(self.history_responses) >= self.max_steps and terminal is None:
            logger.warning(f"Reached maximum steps {self.max_steps}. Forcing termination.")
            terminal = "FAIL"
            other_info["code"] = "FAIL"

        other_info["parsed_responses"] = parsed_responses
        other_info["step_index"] = step_index

        if terminal is not None:
            return prediction, [terminal], other_info

        # 原逻辑：如果超过 max_trajectory_length，默认 FAIL
        if len(self.history_responses) >= self.max_trajectory_length:
            return prediction, ["FAIL"], other_info

        return prediction, actions, other_info

    # -------------------------
    # 补齐：record_step_outcome / summarize（你点名要的）
    # -------------------------
    def record_step_outcome(self, next_obs: Dict):
        """
        与 qwen3vl_agent_local.py / OpenCUAAgentLocal.record_step_outcome 对齐：
        环境执行完当前 action 拿到 next_obs 后调用，补齐 reward prompt 的 “next observation（step_{i+1}.png）”。
        """
        if not self.reward_trajectory:
            logger.warning("record_step_outcome called but reward_trajectory is empty.")
            return

        last = self.reward_trajectory[-1]
        step_index = last.get("step_index")
        if step_index is None:
            step_index = max(0, len(self.history_responses) - 1)

        next_image_path = os.path.join(self.example_result_dir, f"step_{step_index + 1}.png")

        # 尽量把 next_obs 的 screenshot 保存下来（确保 path 存在）
        nb = next_obs.get("screenshot")
        if isinstance(nb, str):
            try:
                nb = base64.b64decode(nb)
            except Exception:
                nb = None
        if isinstance(nb, (bytes, bytearray)):
            _save_bytes_png_if_missing(bytes(nb), next_image_path)

        reward_messages = last.get("reward_messages", [])
        if not reward_messages:
            logger.warning("Last reward_trajectory has empty reward_messages.")
            return

        last_msg = reward_messages[-1]
        content = last_msg.get("content")
        if not isinstance(content, list):
            logger.warning("Last reward user message content is not a list; cannot append next obs.")
            return

        content.append({"type": "text", "text": "\nNext observation after executing this action:\n"})
        content.append({"type": "image", "image": next_image_path})
        content.append({"type": "text", "text": REWARD_INSTRUTION_TEMPLATE_POST})

        last["next_obs_path"] = next_image_path
        if self.trajectory:
            self.trajectory[-1]["next_obs_path"] = next_image_path

    def summarize(self, result) -> str:
        """
        与 qwen3vl_agent_local.py / OpenCUAAgentLocal.summarize 对齐：
        保存 trajectory.json 到 example_result_dir
        """
        out_dir = self.example_result_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "trajectory.json")

        payload = {
            "meta": {"result": result},
            "trajectory": self.trajectory,
            "reward_trajectory": self.reward_trajectory,
        }

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, out_path)
        return out_path
