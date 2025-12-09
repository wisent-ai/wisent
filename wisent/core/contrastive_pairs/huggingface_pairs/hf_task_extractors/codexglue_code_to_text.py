from __future__ import annotations

from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.codexglue_code_to_text_python import CodexglueCodeToTextPythonExtractor

__all__ = ["CodexglueCodeToTextExtractor"]


class CodexglueCodeToTextExtractor(CodexglueCodeToTextPythonExtractor):
    """
    Extractor for CodeXGLUE code-to-text benchmark (default: Python).
    
    This is an alias for the Python variant. For other languages,
    use the specific extractors (codexglue_code_to_text_java, etc.)
    """
    pass
