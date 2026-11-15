from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import requests

from schemas import (
    StructuredAction,
    generation_json_schema,
    legacy_generation_json_schema,
)

logger = logging.getLogger(__name__)

BackendType = Literal["vllm_api", "vllm_offline", "hf_outlines", "ollama"]


@dataclass
class QwenClientConfig:
    model_name_or_path: str = "qwen2.5-coder:7b-instruct"
    lora_adapter_path: Optional[str] = None
    backend: BackendType = "ollama"
    # Guided decoding toggle (disable for a raw schema-validity measurement)
    enable_guided_decoding: bool = False
    # "schema": constrain to the StructuredAction JSON schema (Ollama >= 0.5).
    # "json": any syntactically valid JSON (old behaviour; does not guarantee fields).
    ollama_format: Literal["schema", "json"] = "schema"
    # "commands": current schema (list of commands). "legacy": single utility/args at top
    # level, matching the QLoRA training targets in termguard_sft.jsonl.
    action_format: Literal["commands", "legacy"] = "commands"
    # Generation hyperparameters (deterministic for shell synthesis)
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    seed: int = 0
    # API server settings (vLLM: http://localhost:8000/v1, Ollama: http://localhost:11434)
    api_base_url: str = "http://localhost:11434"
    api_key: str = "termguard-local"
    timeout_sec: float = 60.0
    # Hardware & quantization settings (for vllm_offline / hf_outlines)
    load_in_4bit: bool = True
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85


class QwenStructuredClient:
    """
    Inference client for Qwen2.5-Coder models supporting Ollama, vLLM (API/Offline),
    and HuggingFace Outlines with grammar-constrained JSON schema decoding.
    """

    SYSTEM_PROMPT = (
        "You are TermGuard-Agent, a safety-critical Linux shell copilot. "
        "You must output strictly valid JSON adhering to the provided StructuredAction schema. "
        "Never emit markdown code fences, prose explanations outside JSON, or raw shell strings."
    )

    def __init__(self, config: Optional[QwenClientConfig] = None):
        self.config = config or QwenClientConfig()
        # All fields required for decoding (see schemas.generation_json_schema).
        self.json_schema: Dict[str, Any] = (
            legacy_generation_json_schema()
            if self.config.action_format == "legacy"
            else generation_json_schema()
        )

        self._openai_client = None
        self._vllm_engine = None
        self._vllm_tokenizer = None
        self._lora_request = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._outlines_model = None

        self._initialize_backend()

    def describe(self) -> Dict[str, Any]:
        """For the experiment manifest."""
        c = self.config
        return {
            "model": c.model_name_or_path,
            "backend": c.backend,
            "lora": c.lora_adapter_path,
            "guided_decoding": c.enable_guided_decoding,
            "ollama_format": c.ollama_format,
            "action_format": c.action_format,
            "temperature": c.temperature,
            "top_p": c.top_p,
            "seed": c.seed,
            "max_new_tokens": c.max_new_tokens,
        }

    # ------------------------------------------------------------------ init
    def _initialize_backend(self) -> None:
        if self.config.backend == "ollama":
            pass  # HTTP only
        elif self.config.backend == "vllm_api":
            from openai import OpenAI

            self._openai_client = OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout_sec,
            )
        elif self.config.backend == "vllm_offline":
            self._init_vllm_offline()
        elif self.config.backend == "hf_outlines":
            self._init_hf_outlines()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _init_vllm_offline(self) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self._vllm_tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )
        self._vllm_engine = LLM(
            model=self.config.model_name_or_path,
            dtype="float16",
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enable_lora=bool(self.config.lora_adapter_path),
            max_model_len=8192,
            trust_remote_code=True,
            seed=self.config.seed,
        )
        if self.config.lora_adapter_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest(
                "termguard", 1, self.config.lora_adapter_path
            )

    def _init_hf_outlines(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=bnb_config,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.config.lora_adapter_path:
            from peft import PeftModel

            base_model = PeftModel.from_pretrained(
                base_model, self.config.lora_adapter_path
            )

        base_model.eval()
        self._hf_model = base_model

        if self.config.enable_guided_decoding:
            import outlines

            self._outlines_model = outlines.from_transformers(
                self._hf_model, self._hf_tokenizer
            )

    # ------------------------------------------------------------------ prompts
    def _messages(self, user_prompt: str):
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _format_chat_prompt(self, user_prompt: str) -> str:
        """Applies the Qwen2.5 ChatML template (<|im_start|> / <|im_end|>)."""
        tokenizer = self._vllm_tokenizer or self._hf_tokenizer
        if tokenizer is not None:
            return tokenizer.apply_chat_template(
                self._messages(user_prompt), tokenize=False, add_generation_prompt=True
            )
        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # ------------------------------------------------------------------ generate
    def generate(self, prompt: str) -> str:
        """Callable[[str], str] for TermGuardOrchestrator. Returns a JSON string."""
        if self.config.backend == "ollama":
            return self._generate_ollama(prompt)
        if self.config.backend == "vllm_api":
            return self._generate_vllm_api(prompt)
        if self.config.backend == "vllm_offline":
            return self._generate_vllm_offline(prompt)
        if self.config.backend == "hf_outlines":
            return self._generate_hf_outlines(prompt)
        raise RuntimeError(f"Invalid backend state: {self.config.backend}")

    def _generate_ollama(self, prompt: str) -> str:
        url = f"{self.config.api_base_url.rstrip('/')}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.config.model_name_or_path,
            "messages": self._messages(prompt),
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "seed": self.config.seed,
                "num_predict": self.config.max_new_tokens,
            },
            "stream": False,
        }
        if self.config.enable_guided_decoding:
            # A JSON schema object constrains decoding to the StructuredAction fields;
            # the string "json" only guarantees syntactically valid JSON.
            payload["format"] = (
                self.json_schema if self.config.ollama_format == "schema" else "json"
            )

        response = requests.post(url, json=payload, timeout=self.config.timeout_sec)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "{}").strip()

    def _generate_vllm_api(self, prompt: str) -> str:
        kwargs: Dict[str, Any] = dict(
            model=self.config.model_name_or_path,
            messages=self._messages(prompt),
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            seed=self.config.seed,
        )
        if self.config.enable_guided_decoding:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "StructuredAction", "schema": self.json_schema},
            }
        resp = self._openai_client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "{}").strip()

    def _generate_vllm_offline(self, prompt: str) -> str:
        from vllm import SamplingParams

        sp_kwargs: Dict[str, Any] = dict(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            seed=self.config.seed,
        )
        if self.config.enable_guided_decoding:
            try:  # vLLM <= 0.10
                from vllm.sampling_params import GuidedDecodingParams

                sp_kwargs["guided_decoding"] = GuidedDecodingParams(
                    json=self.json_schema
                )
            except ImportError:  # newer vLLM
                from vllm.sampling_params import StructuredOutputsParams

                sp_kwargs["structured_outputs"] = StructuredOutputsParams(
                    json=self.json_schema
                )

        outputs = self._vllm_engine.generate(
            [self._format_chat_prompt(prompt)],
            SamplingParams(**sp_kwargs),
            lora_request=self._lora_request,
        )
        return outputs[0].outputs[0].text.strip()

    def _generate_hf_outlines(self, prompt: str) -> str:
        formatted_prompt = self._format_chat_prompt(prompt)

        if self.config.enable_guided_decoding and self._outlines_model is not None:
            result = self._outlines_model(
                formatted_prompt,
                StructuredAction,
                max_new_tokens=self.config.max_new_tokens,
            )
            return result if isinstance(result, str) else json.dumps(result)

        import torch

        inputs = self._hf_tokenizer(formatted_prompt, return_tensors="pt").to(
            self._hf_model.device
        )
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self._hf_tokenizer.eos_token_id,
        )
        if self.config.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        else:
            gen_kwargs.update(do_sample=False)  # greedy; no temperature/top_p warnings

        with torch.no_grad():
            outputs = self._hf_model.generate(**inputs, **gen_kwargs)
        response = self._hf_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return response.strip()
