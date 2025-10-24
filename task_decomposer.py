from vlmx.agent import Agent
from typing import Dict, Any
from vlmx.utils import save_json
import os
import json


SYSTEM_INSTRUCTION = """
You are a helpful robotics planning assistant.

Task: Given a high-level instruction, decompose it into a concise ordered list of executable subtasks for a robot.

Guidelines:
- Use clear, imperative phrasing (e.g., "pick up the toy").
- Keep steps minimal and non-overlapping (2-6 items typically).
- Do not include preconditions, alternatives, or commentary; only the steps.
- Each subtask should be a single action and/or a single object.
- Return strictly valid JSON.

Output JSON schema:
```json
{
  "subtasks": ["step one", "step two", "..."]
}
```

For example, if the task is "Put the mug in the basket", the output should be:
{
  "subtasks": ["pick up the mug", "put in the basket"]
}

IMPORTANT: We will use your subtask to guide the robot to focus on the right visual features in the environment. Therefore,
note that in the example, in the first subtask, we only need to focus on the mug, and in the second subtask, we only need to focus on the basket (
not the mug since it's already in the robot's grasp). Generally, each subtask should only focus on one object and/or action.
"""


class TaskDecomposer(Agent):
    OUT_RESULT_PATH = "decomposition.json"

    def _make_system_instruction(self):
        return SYSTEM_INSTRUCTION

    def _make_prompt_parts(self, task_description: str):
        assert isinstance(task_description, str) and task_description.strip(), "task_description must be a non-empty string"
        return [
            "High-level task:",
            task_description.strip(),
            "Produce only the JSON as specified.",
        ]

    def parse_response(self, response, *args, **kwargs) -> Dict[str, Any]:
        text = getattr(response, "text", str(response))
        text = text.strip()
        # Extract JSON by slicing between first '{' and last '}' to avoid markdown fences
        start = text.find("{")
        end = text.rfind("}")
        assert start != -1 and end != -1 and end > start, "Response does not contain valid JSON object"
        json_str = text[start : end + 1]
        parsed = json.loads(json_str, strict=False)
        assert isinstance(parsed, dict) and "subtasks" in parsed, "Parsed JSON must contain 'subtasks' key"
        assert isinstance(parsed["subtasks"], list), "'subtasks' must be a list"
        # Enforce list of strings
        assert all(isinstance(x, str) and x.strip() for x in parsed["subtasks"]), "All subtasks must be non-empty strings"
        save_json(parsed, os.path.join(self.cfg.out_dir, self.OUT_RESULT_PATH))
        return parsed



