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
        # Initialize the Llama model
        self.llm = Llama.from_pretrained(
            repo_id="mradermacher/Reformed-Baptist-1689-Bible-Expert-v3.0-12B-i1-GGUF",
            filename="Reformed-Baptist-1689-Bible-Expert-v3.0-12B.i1-Q4_K_M.gguf",
            n_ctx=4096,  # context window
            n_threads=8,  # adjust based on your CPU
            n_gpu_layers=0,  # set >0 if you have CUDA build
        )

        # Set default generation parameters
        self.default_params = {
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
        }

        self.ready = True
        return self.ready

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
