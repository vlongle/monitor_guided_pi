#!/usr/bin/env python3
"""
Qwen VLM Service for Robot Task Monitoring
Runs in qwen-vlm conda environment and provides HTTP API for image+question processing
"""

import os
import sys
import time
import base64
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

# Flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS

# Image processing
import cv2
import numpy as np
from PIL import Image
import io

# Qwen model imports
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('qwen_service.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QwenConfig:
    """Configuration for Qwen VLM service"""
    model_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    # model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    device: str = "auto"
    max_new_tokens: int = 5  # Reduced for faster inference (just need YES/NO)
    do_sample: bool = True  # Enable sampling for more varied responses
    temperature: float = 0.1 
    port: int = 8766
    host: str = "0.0.0.0"

class QwenVLMService:
    """Main service class for Qwen VLM operations"""
    
    def __init__(self, config: QwenConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.is_ready = False
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for cross-origin requests
        
        # Register routes
        self._register_routes()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen2.5-VL model and processor"""
        try:
            logger.info("Loading Qwen2.5-VL model and processor...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.config.model_path)
            logger.info("✓ Processor loaded successfully")
            
            # Load model
            if "2.5" in self.config.model_path:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16,
                    device_map=self.config.device
                )
            elif "3" in self.config.model_path:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )

            logger.info("✓ Model loaded successfully")
            
            self.is_ready = True
            logger.info("🎉 Qwen VLM service is ready!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_ready = False
    
    def _register_routes(self):
        """Register Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy" if self.is_ready else "unhealthy",
                "model_loaded": self.is_ready,
                "timestamp": time.time()
            })
        
        @self.app.route('/process_image', methods=['POST'])
        def process_image():
            """Process image with question and return answer"""
            try:
                if not self.is_ready:
                    return jsonify({"error": "Model not ready"}), 503
                
                # Parse request
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400
                
                # Extract parameters
                image_data = data.get('image')
                question = data.get('question')
                camera_mode = data.get('camera_mode', 'left_only')  # 'left_only' or 'both'
                
                if not image_data or not question:
                    return jsonify({"error": "Missing image or question"}), 400
                
                # Process image(s)
                if camera_mode == 'left_only':
                    # Single image (left camera)
                    if isinstance(image_data, list) and len(image_data) == 1:
                        image_data = image_data[0]
                    elif isinstance(image_data, list):
                        return jsonify({"error": "Expected single image for left_only mode"}), 400
                    
                    result = self._process_single_image(image_data, question)
                elif camera_mode == 'both':
                    # Two images (left + wrist camera)
                    if not isinstance(image_data, list) or len(image_data) != 2:
                        return jsonify({"error": "Expected list of 2 images for both mode"}), 400
                    
                    result = self._process_dual_images(image_data, question)
                else:
                    return jsonify({"error": "Invalid camera_mode. Use 'left_only' or 'both'"}), 400
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """Get service status"""
            return jsonify({
                "service": "Qwen VLM Service",
                "model_loaded": self.is_ready,
                "model_path": self.config.model_path,
                "device": self.config.device,
                "uptime": time.time() - getattr(self, '_start_time', time.time())
            })
    
    def _decode_image(self, image_data: str) -> np.ndarray:
        """Decode base64 image data to numpy array"""
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array (RGB format)
            image_array = np.array(image)
            
            # Convert BGR to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    def _process_single_image(self, image_data: str, question: str) -> Dict:
        """Process single image with question"""
        # Decode image
        image_array = self._decode_image(image_data)
        pil_image = Image.fromarray(image_array)
        
        # Format question for Qwen
        formatted_question = self._format_question(question)
        
        # Run inference
        start_time = time.time()
        response = self._run_inference(pil_image, formatted_question)
        inference_time = time.time() - start_time
        
        return {
            "success": True,
            "response": response,
            "inference_time": inference_time,
            "camera_mode": "left_only",
            "image_shape": image_array.shape,
            "timestamp": time.time()
        }
    
    def _process_dual_images(self, image_data: List[str], question: str) -> Dict:
        """Process two images (left + wrist) with question"""
        # Decode both images
        left_image = self._decode_image(image_data[0])
        wrist_image = self._decode_image(image_data[1])
        
        left_pil = Image.fromarray(left_image)
        wrist_pil = Image.fromarray(wrist_image)
        
        # Format question for Qwen
        formatted_question = self._format_question(question)
        
        # Run inference with both images
        start_time = time.time()
        response = self._run_inference_dual(left_pil, wrist_pil, formatted_question)
        inference_time = time.time() - start_time
        
        return {
            "success": True,
            "response": response,
            "inference_time": inference_time,
            "camera_mode": "both",
            "left_image_shape": left_image.shape,
            "wrist_image_shape": wrist_image.shape,
            "timestamp": time.time()
        }
    
    def _format_question(self, raw_instruction: str) -> str:
        """Convert raw robot instruction to structured question for Qwen"""
        # Clean up the instruction
        instruction = raw_instruction.strip().lower()
        
        # Add task-specific completion criteria
        task_guidance = ""
        if "put" in instruction and ("in" in instruction or "into" in instruction):
            # Object placement tasks
            if "basket" in instruction or "bowl" in instruction or "container" in instruction:
                task_guidance = "The task is completed when the object is clearly inside the target container, even if not perfectly centered. Look for the object resting within the boundaries of the container."
            elif "on" in instruction:
                task_guidance = "The task is completed when the object is resting on top of the target surface, making contact and stable."
        elif "pick" in instruction or "grab" in instruction or "grasp" in instruction:
            task_guidance = "The task is completed when the robot gripper is clearly holding/grasping the target object."
        elif "open" in instruction:
            task_guidance = "The task is completed when the target object (door, drawer, etc.) is visibly in an open position."
        elif "close" in instruction:
            task_guidance = "The task is completed when the target object is visibly in a closed position."
        else:
            task_guidance = "Look for clear visual evidence that the stated goal has been achieved."
        
        # Format as a clear yes/no question
        formatted = f"""Task: {instruction}

COMPLETION CRITERIA: {task_guidance}

Question: Based on what you see in the camera images, has this task been successfully completed?

Instructions:
- You are analyzing robot manipulation task completion
- You have images from side camera and wrist camera perspectives
- Look carefully at BOTH camera views for evidence of task completion
- Be practical and tolerant - minor imperfections are acceptable if the main goal is achieved
- For placement tasks: the object should be clearly inside/on the target location
- For grasping tasks: the gripper should be holding the object
- Answer ONLY with "YES" if you can clearly see the task is completed
- Answer ONLY with "NO" if the task is not yet completed or unclear

Answer: """
        
        return formatted
    
    def _run_inference(self, image: Image.Image, question: str) -> str:
        """Run inference with single image"""
        try:
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,  # Enable key-value cache for faster generation
                    num_beams=1      # Use greedy search for speed
                )
            
            # Decode response
            generated_text = self.processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise RuntimeError(f"Inference failed: {e}")
    
    def _run_inference_dual(self, left_image: Image.Image, wrist_image: Image.Image, question: str) -> str:
        """Run inference with two images (left + wrist)"""
        try:
            # Prepare messages with both images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": left_image},
                        {"type": "image", "image": wrist_image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=text,
                images=[left_image, wrist_image],
                return_tensors="pt"
            )
            
            # Move inputs to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,  # Enable key-value cache for faster generation
                    num_beams=1      # Use greedy search for speed
                )
            
            # Decode response
            generated_text = self.processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Dual image inference error: {e}")
            raise RuntimeError(f"Dual image inference failed: {e}")
    
    def run(self):
        """Run the Flask service"""
        self._start_time = time.time()
        logger.info(f"Starting Qwen VLM service on {self.config.host}:{self.config.port}")
        
        try:
            self.app.run(
                host=self.config.host,
                port=self.config.port,
                debug=False,
                threaded=True
            )
        except Exception as e:
            logger.error(f"Service failed to start: {e}")
            raise

def main():
    """Main entry point"""
    # Configuration
    config = QwenConfig()
    
    # Create and run service
    service = QwenVLMService(config)
    
    if service.is_ready:
        service.run()
    else:
        logger.error("Service failed to initialize")
        sys.exit(1)

if __name__ == "__main__":
    main()
