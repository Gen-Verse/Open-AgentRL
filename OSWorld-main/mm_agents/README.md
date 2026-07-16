# Agent
## Prompt-based Agents

### Supported Models
We currently support the following models as the foundational models for the agents:
- `GPT-3.5` (gpt-3.5-turbo-16k, ...)
- `GPT-4` (gpt-4-0125-preview, gpt-4-1106-preview, ...)
- `GPT-4V` (gpt-4-vision-preview, ...)
- `Gemini-Pro`
- `Gemini-Pro-Vision`
- `Claude-3, 2` (claude-3-haiku-2024030, claude-3-sonnet-2024022, ...)
- ...

And those from the open-source community:
- `Mixtral 8x7B`
- `QWEN`, `QWEN-VL`
- `CogAgent`
- `Llama3`
- `MiniMax-M3`
- `MiniMax-M2.7` (text-only)
- ...

In the future, we will integrate and support more foundational models to enhance digital agents, so stay tuned.

### Remote Chat Configuration

The prompt agent can use the two supported models through either compatible chat protocol. Set the API key and choose a region and protocol before running an agent:

```bash
export MINIMAX_API_KEY="your_api_key"
export MINIMAX_REGION="global_en"  # or cn_zh
export MINIMAX_PROTOCOL="openai"    # or anthropic
python run.py --model MiniMax-M3 --observation_type screenshot
```

The region and protocol combinations select these base URLs:

| Region | OpenAI-compatible | Anthropic-compatible |
| --- | --- | --- |
| `global_en` | `https://api.minimax.io/v1` | `https://api.minimax.io/anthropic` |
| `cn_zh` | `https://api.minimaxi.com/v1` | `https://api.minimaxi.com/anthropic` |

`MiniMax-M3` accepts text and image observations. `MiniMax-M2.7` accepts text observations, so use `--observation_type a11y_tree` with that model. To explicitly control M3 reasoning, set `MINIMAX_THINKING` to `adaptive` or `disabled`; M2.7 reasoning remains enabled by the service.

### How to use

```python
from mm_agents.agent import PromptAgent

agent = PromptAgent(
    model="gpt-4-vision-preview",
    observation_type="screenshot",
)
agent.reset()
# say we have an instruction and observation
instruction = "Please help me to find the nearest restaurant."
obs = {"screenshot": open("path/to/observation.jpg", 'rb').read()}
response, actions = agent.predict(
    instruction,
    obs
)
```

### Observation Space and Action Space
We currently support the following observation spaces:
- `a11y_tree`: the accessibility tree of the current screen
- `screenshot`: a screenshot of the current screen
- `screenshot_a11y_tree`: a screenshot of the current screen with the accessibility tree overlay
- `som`: the set-of-mark trick on the current screen, with table metadata included.

And the following action spaces:
- `pyautogui`: valid Python code with `pyautogui` code valid
- `computer_13`: a set of enumerated actions designed by us

To feed an observation into the agent, you have to maintain the `obs` variable as a dict with the corresponding information:
```python
# continue from the previous code snippet
obs = {
    "screenshot": open("path/to/observation.jpg", 'rb').read(),
    "a11y_tree": ""  # [a11y_tree data]
}
response, actions = agent.predict(
    instruction,
    obs
)
```

## Efficient Agents, Q* Agents, and more
Stay tuned for more updates.
