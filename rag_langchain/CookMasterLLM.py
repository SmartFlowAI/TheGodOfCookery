from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from modelscope import AutoModelForCausalLM, AutoTokenizer


class CookMasterLLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM2"
