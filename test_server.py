#!/usr/bin/env python3
"""
Minimal client test for server.py /predict endpoint sending numpy arrays.
"""

import requests
import numpy as np


def main():
    base_url = "http://localhost:8766"

    # Health
    r = requests.get(f"{base_url}/health", timeout=5)
    print("health:", r.status_code, r.json())

    # Build a simple trajectory with two camera views
    h, w = 240, 320
    left = np.zeros((h, w, 3), dtype=np.uint8)
    left[:] = (120, 160, 200)
    wrist = np.zeros((h, w, 3), dtype=np.uint8)
    wrist[:] = (30, 60, 90)

    trajectory = {
        "1": {
            "left": left.tolist(),
            "wrist": wrist.tolist(),
        }
    }

    payload = {
        "task_description": "Put the mug in the basket",
        "trajectory": trajectory,
    }

    r = requests.post(f"{base_url}/predict", json=payload, timeout=120)
    print("predict:", r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)


if __name__ == "__main__":
    main()


