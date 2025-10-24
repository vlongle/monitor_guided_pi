from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    "torch",
    "torchvision",
    "transformers",
    "flask",
    "flask-cors",
    "qwen_vl_utils",
    "accelerate",
]




setup(
    name='monitor_guided_pi',
    version='0.1.0',
    description='Monitor Guided PI',
    install_requires=INSTALL_REQUIRES, 
    packages=find_packages(),
    python_requires='>=3.11',
)