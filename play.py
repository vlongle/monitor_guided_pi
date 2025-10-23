
from vlmx.utils import *
import matplotlib.pyplot as plt
from vlm_monitor import VLMMonitor, AgentConfig

video="videos/four_view_put_the_mug_in_the_basket_2025_10_22_11:00:21.mp4"

trajectory = extract_camera_views_from_video(video)

# t=100
t=500
cam_views = trajectory[t]

# cam_views_to_include = ['left', 'right', 'wrist', 'overhead']
cam_views_to_include = ['left', 'wrist']


cam_views = {
    k: v for k, v in cam_views.items() if k in cam_views_to_include
}

API_KEY="AIzaSyAIaEC3qr4tw1jilS7cwtjEf6KyefYIZVo"
agent = VLMMonitor(AgentConfig(
        # model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        # model_name="Qwen/Qwen2.5-VL-14B-Instruct",
        # model_name="Qwen/Qwen3-VL-8B-Instruct",
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        # model_name="gemini-2.5-flash",
        out_dir="test_results",
        api_key=API_KEY
    ))

# task_description = "Put the mug in the basket"
task_description = "Pick up the mug"
trajectories = {
    1: cam_views,
}
agent.generate_prediction(trajectories, task_description)
