from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["NoticiaExtractor"]
_LOG = setup_logger(__name__)

task_names = ("noticia",)

class NoticiaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Noticia benchmark."""

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Noticia doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Noticia format (Spanish clickbait summarization):
        - web_headline: the sensationalist/clickbait headline
        - web_text: the actual news article body
        - summary: the target one-sentence summary revealing the truth
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Noticia format
            if "web_headline" in doc and "web_text" in doc and "summary" in doc:
                headline = str(doc.get("web_headline", "")).strip()
                text = str(doc.get("web_text", "")).strip()
                summary = str(doc.get("summary", "")).strip()

                if not headline or not text or not summary:
                    log.debug("Skipping doc with missing headline/text/summary", extra={"doc": doc})
                    return None

                # Prompt is the instruction + headline + text (as in lm-eval)
                prompt = f"Ahora eres una Inteligencia Artificial experta en desmontar titulares sensacionalistas o clickbait. Tu tarea consiste en analizar noticias con titulares sensacionalistas y generar un resumen de una sola frase que revele la verdad detrás del titular.\nEste es el titular de la noticia: {headline}\nEl titular plantea una pregunta o proporciona información incompleta. Debes buscar en el cuerpo de la noticia una frase que responda lo que se sugiere en el título. Siempre que puedas cita el texto original, especialmente si se trata de una frase que alguien ha dicho. Si citas una frase que alguien ha dicho, usa comillas para indicar que es una cita. Usa siempre las mínimas palabras posibles. No es necesario que la respuesta sea una oración completa, puede ser sólo el foco de la pregunta. Recuerda responder siempre en Español.\nEste es el cuerpo de la noticia:\n{text}"

                # Positive: the actual summary
                correct = summary

                # Negative: generic refusal (similar to other summarization tasks)
                incorrect = "No puedo proporcionar un resumen de esta noticia."

                metadata = {"label": "noticia"}

                return self._build_pair(
                    question=prompt,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            log.debug("Skipping doc without web_headline/web_text/summary fields", extra={"doc": doc})
            return None

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
