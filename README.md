# TermGuard-Agent 🛡️

**TermGuard-Agent** is a safety-critical, multi-stage Linux shell copilot designed to bridge natural language requests and Unix command execution. It mitigates catastrophic command execution risks in untrusted command-line environments using a fail-closed architecture, deterministic AST policy verification, and Docker-based sandboxing.

---

## 🏗 Architecture Overview

TermGuard-Agent operates on a dual-layer defensive pipeline:
1. **Structured Neural Generation:** Utilizes Qwen2.5-Coder via Ollama to generate strict `List[CommandNode]` JSON arrays (validated via Pydantic).
2. **Deterministic Static Policy Gate:** Analyzes the generated commands using `tree-sitter-bash` to enforce path normalization, directory confinement, and hard-blocks on dangerous shell constructs (like subshells and obfuscation).
3. **Hardened Container Sandbox:** Executes validated commands within an isolated `ubuntu:24.04` Docker container running as a non-root user with dropped capabilities, read-only root filesystems, and strict PID constraints.

---

## 📋 Prerequisites

Before setting up the project, ensure you have the following installed on your system:
* **Python 3.10+**
* **Docker Engine** (Must be running and accessible by your user)
* **Ollama** (For local LLM inference)

---

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/gokulc01/termgaurd.git
cd termgaurd
```

**2. Install Python Dependencies**
```bash
pip install -r requirements.txt
```
*(Dependencies include `pydantic`, `docker`, `tree-sitter`, `tree-sitter-bash`, and `requests`)*

**3. Prepare the Docker Sandbox Environment**
Pull the required Ubuntu image that the sandbox runner uses to execute commands safely:
```bash
docker pull ubuntu:24.04
```

**4. Setup the Local LLM via Ollama**
Pull the Qwen2.5 Coder 3B instruction-tuned model. This serves as the baseline generation engine for TermGuard-Agent.
```bash
ollama pull qwen2.5-coder:3b-instruct
```
*(Ensure the Ollama server is running locally on `http://localhost:11434`)*

---

## 💻 Usage

### 1. Interactive CLI (Verbose Mode)
You can use the verbose CLI to pass natural language prompts to TermGuard-Agent. It will display the entire lifecycle: LLM generation, AST Gate evaluation, and the final synthesized command.

```bash
python cli_inline_verbose.py "create directory src/ and enter the directory"
```
If the command passes all safety checks, the synthesized safe bash sequence (e.g., `mkdir src && cd src`) will be output to standard out.

### 2. Running the Evaluation Benchmark Suite
The project includes a 60-task benchmark evaluating the agent against benign administration, catastrophic intent, and adversarial scope tasks. 

To run the ablation matrix (C0 - C3) and evaluate the model's Unsafe Execution Rate (UER) and Benign Completion Rate (BCR):
```bash
python run_benchmark.py --output final_benchmark_results.json
```
If you encounter Docker 409 conflict errors, ensure all orphaned containers are cleared:
```bash
docker ps -a --filter "ancestor=ubuntu:24.04" -q | xargs -r docker rm -f
```

---

## 📁 Core Project Structure
* `schemas.py`: Defines Pydantic validation structures, including the `CommandNode` array for multi-line execution.
* `static_gate.py`: Implements the `tree-sitter-bash` AST parser to enforce fail-closed security policies.
* `sandbox_runner.py`: Manages the hardened Docker container lifecycle and executes commands with strict resource limits.
* `orchestrator.py`: Coordinates the flow between the LLM, RAG retriever, static gate, and sandbox runner.
* `llm_client.py`: Handles HTTP interfacing with the Ollama inference backend.
* `cli_inline_verbose.py`: The main entry point for natural language command generation.

---
**Authors:** Giridharan, Gokul, Radha Krishna, Sriram | LMA Monsoon 2026