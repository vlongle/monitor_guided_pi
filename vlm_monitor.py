from vlmx.agent import Agent, AgentConfig
from typing import List, Dict
import numpy as np
from vlmx.utils import save_json
from dotenv import load_dotenv
import os
import json
from PIL import Image
# Load API key from .env file
load_dotenv()
API_KEY = os.environ.get('API_KEY')


SYSTEM_INSTRUCTION = """
You are a robotic monitor system. 

## Input
At each time step, you will shown images from different camera views (e.g., "left" and "right" and "overhead" cameras of the scene and "wrist" camera from the robot's end-effector).
You might also be shown frames from k time steps ago to provide context for the current time step. Then, you're also given a task description in text.
When we provide multiple time steps, we will also provide the "frame: {frame_idx}" for each time steps, where lower frame_idx means more distant time step in the past, and highest frame_idx
in the sequence is the most recent time step.

## Task
Your task is to examine all inputs, and determine the progress of the task i.e., whether the task is completed or not.


## Output

Your output should be a JSON as follows:

```json
{
    "detailed_description": {
    "frame {frame_idx}": "description of the frame",
    ...
    }
    "reasoning": "Your reasoning for the progress" # Your reasoning for the progress
    "completed": true/false # Whether the task is completed or not
}
```
Make sure that the result is a valid JSON.
"""

class VLMMonitor(Agent):
    OUT_RESULT_PATH = "output.json"

    def _make_system_instruction(self):
        return SYSTEM_INSTRUCTION

    def _make_prompt_parts(self, trajectory: Dict[int, Dict[str, np.ndarray]], task_description: str):
        prompt_parts = [
            "Here is the task description:",
            task_description,
            "Here are the camera views:",
        ]
        
        # Convert numpy arrays to PIL Images and add them to prompt
        for frame_idx, cam_views in trajectory.items():
            prompt_parts.append(f"Frame: {frame_idx}")
            for cam_name, img_array in cam_views.items():
                prompt_parts.append(f"Camera: {cam_name}")
                # Convert numpy array to PIL Image
                if isinstance(img_array, np.ndarray):
                    # Ensure the array is in the right format (uint8)
                    if img_array.dtype != np.uint8:
                        img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
                    pil_image = Image.fromarray(img_array)
                    prompt_parts.append(pil_image)
                elif isinstance(img_array, Image.Image):
                    # Already a PIL Image
                    prompt_parts.append(img_array)
                else:
                    raise ValueError(f"Unsupported image type: {type(img_array)}")
            
        print("Done making prompt parts")
        return prompt_parts

    def parse_response(self, response):
        json_str = response.text.strip().strip("```json").strip()
        print("json_str:")
        print(json_str)
        parsed_response = json.loads(json_str, strict=False)
        print("parsed_response:")
        print(parsed_response)
        save_json(parsed_response, os.path.join(self.cfg.out_dir, self.OUT_RESULT_PATH))
        return parsed_response


if __name__ == "__main__":
    # Initialize the agent
    agent = VLMMonitor(AgentConfig(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        out_dir="test_results",
        api_key=API_KEY
    ))

    # Generate a prediction
    response = agent.generate_prediction("What's 2+2?")
