"""PolyMath language-specific subclasses: Korean through Chinese."""
from __future__ import annotations

from .polymath import PolyMathExtractor

class PolyMathKOTopExtractor(PolyMathExtractor):
    language = "ko"
    difficulty = "top"

class PolyMathKOHighExtractor(PolyMathExtractor):
    language = "ko"
    difficulty = "high"

class PolyMathKOMediumExtractor(PolyMathExtractor):
    language = "ko"
    difficulty = "medium"

class PolyMathKOLowExtractor(PolyMathExtractor):
    language = "ko"
    difficulty = "low"

# Malay
class PolyMathMSTopExtractor(PolyMathExtractor):
    language = "ms"
    difficulty = "top"

class PolyMathMSHighExtractor(PolyMathExtractor):
    language = "ms"
    difficulty = "high"

class PolyMathMSMediumExtractor(PolyMathExtractor):
    language = "ms"
    difficulty = "medium"

class PolyMathMSLowExtractor(PolyMathExtractor):
    language = "ms"
    difficulty = "low"

# Portuguese
class PolyMathPTTopExtractor(PolyMathExtractor):
    language = "pt"
    difficulty = "top"

class PolyMathPTHighExtractor(PolyMathExtractor):
    language = "pt"
    difficulty = "high"

class PolyMathPTMediumExtractor(PolyMathExtractor):
    language = "pt"
    difficulty = "medium"

class PolyMathPTLowExtractor(PolyMathExtractor):
    language = "pt"
    difficulty = "low"

# Russian
class PolyMathRUTopExtractor(PolyMathExtractor):
    language = "ru"
    difficulty = "top"

class PolyMathRUHighExtractor(PolyMathExtractor):
    language = "ru"
    difficulty = "high"

class PolyMathRUMediumExtractor(PolyMathExtractor):
    language = "ru"
    difficulty = "medium"

class PolyMathRULowExtractor(PolyMathExtractor):
    language = "ru"
    difficulty = "low"

# Swahili
class PolyMathSWTopExtractor(PolyMathExtractor):
    language = "sw"
    difficulty = "top"

class PolyMathSWHighExtractor(PolyMathExtractor):
    language = "sw"
    difficulty = "high"

class PolyMathSWMediumExtractor(PolyMathExtractor):
    language = "sw"
    difficulty = "medium"

class PolyMathSWLowExtractor(PolyMathExtractor):
    language = "sw"
    difficulty = "low"

# Telugu
class PolyMathTETopExtractor(PolyMathExtractor):
    language = "te"
    difficulty = "top"

class PolyMathTEHighExtractor(PolyMathExtractor):
    language = "te"
    difficulty = "high"

class PolyMathTEMediumExtractor(PolyMathExtractor):
    language = "te"
    difficulty = "medium"

class PolyMathTELowExtractor(PolyMathExtractor):
    language = "te"
    difficulty = "low"

# Thai
class PolyMathTHTopExtractor(PolyMathExtractor):
    language = "th"
    difficulty = "top"

class PolyMathTHHighExtractor(PolyMathExtractor):
    language = "th"
    difficulty = "high"

class PolyMathTHMediumExtractor(PolyMathExtractor):
    language = "th"
    difficulty = "medium"

class PolyMathTHLowExtractor(PolyMathExtractor):
    language = "th"
    difficulty = "low"

# Vietnamese
class PolyMathVITopExtractor(PolyMathExtractor):
    language = "vi"
    difficulty = "top"

class PolyMathVIHighExtractor(PolyMathExtractor):
    language = "vi"
    difficulty = "high"

class PolyMathVIMediumExtractor(PolyMathExtractor):
    language = "vi"
    difficulty = "medium"

class PolyMathVILowExtractor(PolyMathExtractor):
    language = "vi"
    difficulty = "low"

# Chinese
class PolyMathZHTopExtractor(PolyMathExtractor):
    language = "zh"
    difficulty = "top"

class PolyMathZHHighExtractor(PolyMathExtractor):
    language = "zh"
    difficulty = "high"

class PolyMathZHMediumExtractor(PolyMathExtractor):
    language = "zh"
    difficulty = "medium"

class PolyMathZHLowExtractor(PolyMathExtractor):
    language = "zh"
    difficulty = "low"
