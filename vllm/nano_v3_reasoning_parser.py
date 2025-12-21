from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.entrypoints.openai.protocol import DeltaMessage


@ReasoningParserManager.register_module("nano_v3")
class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for Nemotron-3-Nano that handles the enable_thinking flag.

    When enable_thinking is False, the model doesn't emit <think> tags, so all
    content comes through as "reasoning" in the parent parser. This class swaps
    the fields so content appears in the correct field for both streaming and
    non-streaming modes.
    """

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        # Capture chat_template_kwargs for use in streaming
        self.chat_template_kwargs = kwargs.get("chat_template_kwargs", {})
        self._thinking_disabled = (
            self.chat_template_kwargs
            and self.chat_template_kwargs.get("enable_thinking") is False
        )

    def extract_reasoning(self, model_output, request):
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )
        if (
            hasattr(request, "chat_template_kwargs")
            and request.chat_template_kwargs
            and request.chat_template_kwargs.get("enable_thinking") is False
            and final_content is None
        ):
            reasoning_content, final_content = final_content, reasoning_content

        return reasoning_content, final_content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: tuple,
        current_token_ids: tuple,
        delta_token_ids: tuple,
    ) -> DeltaMessage | None:
        # Get result from parent class
        result = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        # When thinking is disabled and we get reasoning content, swap to content
        if result is not None and self._thinking_disabled:
            if result.reasoning_content is not None and result.content is None:
                return DeltaMessage(
                    content=result.reasoning_content,
                    reasoning_content=None,
                )

        return result