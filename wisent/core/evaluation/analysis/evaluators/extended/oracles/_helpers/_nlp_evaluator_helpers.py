"""Extracted NLP evaluator helper methods: NLI cross-encoder and embedding similarity."""

from __future__ import annotations


class NLPEvaluatorHelpersMixin:
    """Mixin providing NLI and embedding helper methods for NLPEvaluator."""

    CE_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
    EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def _load_ce(self):
        """Load the NLI cross-encoder model.
        Cross-encoder models are small and load quickly. They run on CPU
        reasonably well. They provide strong performance for entailment tasks.
        """
        from sentence_transformers import CrossEncoder
        _CE = CrossEncoder(self.CE_MODEL_NAME)
        return _CE

    def _nli_pick_between(
        self, response: str, options: list[str],
    ) -> tuple[int | None, list[float], float]:
        """
        Compare entailment(response -> 'The correct option is: <opt_i>')
        for i in {A,B}.
        Returns: (pred_idx, [entA, entB], margin)
        """
        ce = self._load_ce()
        pairs = [(response, f"The correct option is: {opt}") for opt in options]
        import torch
        import torch.nn.functional as F
        logits = torch.tensor(ce.predict(pairs))
        probs = F.softmax(logits, dim=-1).tolist()
        ent = [p[1] for p in probs]
        pred_idx = 1 if ent[0] > ent[1] else 2
        margin = abs(ent[0] - ent[1])
        return pred_idx, ent, margin

    def _nli_entailment_pair(
        self, a: str, bnormalize_text: str,
    ) -> tuple[float | None, float | None]:
        """Entailment probabilities for (a -> b) and (b -> a)."""
        try:
            ce = self._load_ce()
        except Exception:
            return None, None
        pairs = [(a, bnormalize_text), (bnormalize_text, a)]
        import torch
        import torch.nn.functional as F
        logits = torch.tensor(ce.predict(pairs))
        probs = F.softmax(logits, dim=-1).tolist()
        return probs[0][1], probs[1][1]

    def _load_emb(self):
        """Load the sentence embedding model."""
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer(self.EMB_MODEL_NAME)
        return _EMB

    def _emb_sim(self, a: str, b: str) -> float | None:
        """Compute cosine similarity between two texts."""
        try:
            emb = self._load_emb()
        except Exception:
            return None
        import torch
        va, vb = emb.encode(
            [a, b], convert_to_tensor=True, normalize_embeddings=True)
        return torch.matmul(va, vb).item()

    def _emb_sims(
        self, response: str, options: list[str],
    ) -> tuple[float | None, float | None]:
        """Compute cosine similarities between response and two options."""
        try:
            emb = self._load_emb()
        except Exception:
            return None, None
        import torch
        vecs = emb.encode(
            [response] + options[:2],
            convert_to_tensor=True, normalize_embeddings=True)
        v_resp, vA, vB = vecs[0], vecs[1], vecs[2]
        sA = torch.matmul(v_resp, vA).item()
        sB = torch.matmul(v_resp, vB).item()
        return sA, sB
