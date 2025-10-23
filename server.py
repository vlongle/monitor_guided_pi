#!/usr/bin/env python3
"""Minimal VLM Agent HTTP Service using our VLMMonitor agent.

POST /predict with JSON body:
{
  "task_description": str,
  "trajectory": { "1": { "left": [[...],[...],...], "wrist": [[...]] } }
}
Values under cameras should be image arrays (lists); we convert them to np.uint8.
"""

import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np

from vlm_monitor import VLMMonitor, AgentConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    out_dir: str = "service_results"
    api_key: str = os.getenv("VLM_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    host: str = "0.0.0.0"
    port: int = 8766


class VLMServer:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.app = Flask(__name__)
        CORS(self.app)

        # use_verbose_output=False
        use_verbose_output=True

        self.agent = VLMMonitor(AgentConfig(
            model_name=self.config.model_name,
            out_dir=self.config.out_dir,
            api_key=self.config.api_key,
            use_verbose_output=use_verbose_output
        ))
        self.is_ready = True
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.route('/health', methods=['GET'])
        def health() -> Any:
            return jsonify({"status": "healthy" if self.is_ready else "unhealthy"})

        @self.app.route('/predict', methods=['POST'])
        def predict() -> Any:
            data = request.get_json()
            assert data is not None, "No JSON body provided"
            task: str = data.get("task_description")
            trajectory_json: Dict[str, Dict[str, Any]] = data.get("trajectory")
            assert task and trajectory_json, "Missing task_description or trajectory"

            # Convert lists to numpy arrays (uint8) and keys to ints
            trajectory: Dict[int, Dict[str, np.ndarray]] = {}
            for frame_idx_str, cam_views in trajectory_json.items():
                frame_idx = int(frame_idx_str)
                trajectory[frame_idx] = {}
                for cam_name, img_list in cam_views.items():
                    assert isinstance(img_list, list), "Images must be list (array-like)"
                    arr = np.array(img_list, dtype=np.uint8)
                    trajectory[frame_idx][cam_name] = arr

            start = time.time()
            result = self.agent.generate_prediction(trajectory, task, overwrite=True,
            gen_config={"max_new_tokens": 256, "temperature": 0.1}
            # gen_config={"max_new_tokens": 20, "temperature": 0.1}
            )
            return jsonify({
                "success": True,
                "result": result,
                "latency_s": time.time() - start,
            })

    def run(self) -> None:
        self.app.run(host=self.config.host, port=self.config.port, debug=False, threaded=True)


def main() -> None:
    server = VLMServer(ServiceConfig())
    server.run()


if __name__ == "__main__":
    main()


