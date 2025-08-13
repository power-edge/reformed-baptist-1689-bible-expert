# model.py
from typing import List, Dict, Any, Optional
import numpy as np
from mlserver import MLModel
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    RequestInput,
    ResponseOutput,
)
from mlserver.codecs import StringCodec
from llama_cpp import Llama
import json


class LlamaMLServerModel(MLModel):
    """
    MLServer wrapper for llama-cpp-python model
    """

    async def load(self) -> bool:
        """
        Load the Llama model when MLServer starts
        """
        # Get model configuration from settings
        model_config = self._get_model_config()

        # Load from Hugging Face
        self.llm = Llama.from_pretrained(
            repo_id=model_config["repo_id"],
            filename=model_config["filename"],
            **model_config.get("llama_params", {}),
        )

        # Set default generation parameters from config
        self.default_params = model_config.get(
            "generation_params",
            {
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
            },
        )

        self.ready = True
        return self.ready

    def _get_model_config(self) -> Dict[str, Any]:
        """
        Extract model configuration from MLServer settings with deep merge of defaults
        """
        # Default configuration with comprehensive defaults
        default_config = {
            "repo_id": "mradermacher/Reformed-Baptist-1689-Bible-Expert-v3.0-12B-i1-GGUF",
            "filename": "Reformed-Baptist-1689-Bible-Expert-v3.0-12B.i1-Q4_K_M.gguf",
            "llama_params": {
                "n_ctx": 4096,
                "n_threads": 8,
                "n_gpu_layers": 0,
                "n_batch": 512,
                "verbose": False,
                "use_mlock": False,
                "use_mmap": True,
                "rope_freq_base": 10000.0,
                "rope_freq_scale": 1.0,
                "logits_all": False,
                "embedding": False,
                "offload_kqv": True,
                "last_n_tokens_size": 64,
                "seed": -1,
                "f16_kv": True,
                "low_vram": False,
            },
            "generation_params": {
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "tfs_z": 1.0,
                "typical_p": 1.0,
                "mirostat_mode": 0,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "stop": [],
                "seed": -1,
                "logprobs": None,
                "echo": False,
                "suffix": None,
            },
        }

        # Deep merge user configuration with defaults
        if hasattr(self.settings, "parameters") and self.settings.parameters:
            config = self._deep_merge_config(default_config, self.settings.parameters)
            return config

        return default_config

    def _deep_merge_config(
        self, default: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge configuration dictionaries, preserving individual parameter defaults
        """
        result = default.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Deep merge nested dictionaries (llama_params, generation_params)
                result[key] = self._deep_merge_config(result[key], value)
            else:
                # Direct assignment for non-dict values
                result[key] = value

        return result

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """
        Handle inference requests
        """
        # Extract input data
        input_data = self._extract_inputs(payload.inputs)

        # Determine inference type and generate response
        if "messages" in input_data:
            # Chat completion mode
            response_text = self._chat_completion(input_data)
        else:
            # Text completion mode
            response_text = self._text_completion(input_data)

        # Create response
        return self._create_response(response_text, payload.id)

    def _extract_inputs(self, inputs: List[RequestInput]) -> Dict[str, Any]:
        """
        Extract and parse input data from the request
        """
        input_data = {}

        for request_input in inputs:
            if request_input.name == "prompt":
                # Simple text completion
                input_data["prompt"] = StringCodec.decode_input(request_input)[0]

            elif request_input.name == "messages":
                # Chat completion - expect JSON string
                messages_str = StringCodec.decode_input(request_input)[0]
                input_data["messages"] = json.loads(messages_str)

            elif request_input.name == "parameters":
                # Generation parameters
                params_str = StringCodec.decode_input(request_input)[0]
                input_data["parameters"] = json.loads(params_str)

        return input_data

    def _text_completion(self, input_data: Dict[str, Any]) -> str:
        """
        Handle simple text completion
        """
        prompt = input_data.get("prompt", "")
        params = {**self.default_params, **input_data.get("parameters", {})}

        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=params.get("max_tokens", 256),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.95),
            top_k=params.get("top_k", 40),
        )

        return response["choices"][0]["text"]

    def _chat_completion(self, input_data: Dict[str, Any]) -> str:
        """
        Handle chat completion
        """
        messages = input_data.get("messages", [])
        params = {**self.default_params, **input_data.get("parameters", {})}

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=params.get("max_tokens", 256),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.95),
            top_k=params.get("top_k", 40),
        )

        return response["choices"][0]["message"]["content"]

    def _create_response(
        self, text: str, request_id: Optional[str] = None
    ) -> InferenceResponse:
        """
        Create MLServer inference response
        """
        # Encode the text response
        response_output = ResponseOutput(
            name="generated_text",
            shape=[1],
            datatype="BYTES",
            data=StringCodec.encode_output("generated_text", [text]).data,
        )

        return InferenceResponse(
            id=request_id,
            model_name=self.name,
            model_version=self.version,
            outputs=[response_output],
        )
