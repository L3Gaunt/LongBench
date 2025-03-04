# LongBench Development Guide

## Commands
- **Install**: `pip install -r requirements.txt`
- **Model Deployment**: `vllm serve <model_path> --api-key <token> --tensor-parallel-size <num_gpus> --gpu-memory-utilization <value> --max_model_len <context_length> --trust-remote-code`
- **Run Evaluation**: `python pred.py --model <model_name>`
- **Processing Results**: `python result.py`
- **Additional Options**: 
  - `--cot`: Chain-of-Thought reasoning
  - `--no_context`: Test without long context
  - `--rag N`: Use top-N retrieved contexts

## Code Style Guidelines
- **Python Style**: Follow PEP 8 conventions
- **Imports**: Group standard library, third-party, and local imports
- **Naming**: 
  - Use snake_case for variables and functions
  - Use descriptive names that reflect purpose
- **Error Handling**: Use try-except with retries for API calls
- **Documentation**: Include docstrings for functions and classes
- **Configuration**: Use JSON files in config/ directory
- **Type Annotations**: Add type hints where appropriate