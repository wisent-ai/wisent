from __future__ import annotations

from wisent_guard.synthetic.cleaners.core.atoms import ChatMessage, CompletionFn

def model_adapter(model: object, **gen_kwargs) -> CompletionFn:
    """
    Wrap a model exposing `.generate(batch_of_messages, **kw) -> list[str]`.
    We call it with a single-item batch and return the first string.
    """
    def _call(messages: list[ChatMessage]) -> str:
        out = model.generate([messages], use_steering=False, **gen_kwargs)  # type: ignore[attr-defined]
        return out[0] if out else ""
    return _call
