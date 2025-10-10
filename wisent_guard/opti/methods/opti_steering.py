
from __future__ import annotations

import random
from typing import Any, Callable, Sequence

import optuna

from core.atoms import BaseObjective

# Type aliases for clarity
Prompt = list[dict[str, str]]  # chat-format: [{"role":..., "content":...}, ...]
JudgeFn = Callable[[str, str, str], int]  # (question, base, steered) -> 1 if steered wins, else 0


class SteeringActivationObjective(BaseObjective):
    """Optimize (layer, alpha) for activation steering using an external LLM judge.

    Parameters
    ----------
    wm : Any
        WisentModel-like object exposing:
         - num_layers (int)
         - set_steering_from_raw(raw: dict[str, tensor], scale: float, normalize: bool)
         - clear_steering()
         - generate(inputs: list[Prompt], use_steering: bool, **kw) -> list[str]
    val_prompts : list[Prompt]
        Validation set (chat message lists) used for A/B comparisons.
    vectors_by_layer : dict[str|int, Any]
        Mapping of layer name ("1".. "L") or index to steering vector tensor for that layer.
    judge_fn : JudgeFn
        Callable that returns 1 if the steered answer is preferred (happier & not worse), else 0.
    alpha_range : tuple[float, float]
        Lower/upper bounds for the steering strength.
    candidate_layers : Sequence[str|int] | None
        Optional subset of layers to search; default: keys of `vectors_by_layer` filtered to valid.
    sample_size : int
        Number of prompts to sample per trial (keeps cost bounded).
    batch_size : int
        Evaluate in mini-batches; we report win-rate after each batch to enable pruning.
    normalize_vectors : bool
        Whether to L2-normalize vectors before scaling by alpha.
    gen_kwargs : dict
        Extra kwargs forwarded to `wm.generate` (e.g., max_new_tokens).
    """

    name = "steering-activation"
    direction = "maximize"

    def __init__(
        self,
        wm: Any,
        val_prompts: list[Prompt],
        vectors_by_layer: dict[str | int, Any],
        judge_fn: JudgeFn,
        alpha_range: tuple[float, float] = (-3.0, 3.0),
        candidate_layers: Sequence[str | int] | None = None,
        sample_size: int = 64,
        batch_size: int = 16,
        normalize_vectors: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.wm = wm
        self.val_prompts = list(val_prompts)
        self.vectors_by_layer = {str(k): v for k, v in vectors_by_layer.items()}
        L = int(getattr(wm, "num_layers"))
        valid = {str(i) for i in range(1, L + 1)}
        if candidate_layers is None:
            self.candidate_layers = sorted(list(valid.intersection(self.vectors_by_layer.keys())), key=lambda s: int(s))
        else:
            self.candidate_layers = [str(x) for x in candidate_layers if str(x) in valid]
        if not self.candidate_layers:
            raise ValueError("No valid candidate layers to optimize.")
        self.alpha_lo, self.alpha_hi = alpha_range
        self.sample_size = int(sample_size)
        self.batch_size = max(1, int(batch_size))
        self.normalize_vectors = bool(normalize_vectors)
        self.gen_kwargs = dict(gen_kwargs or {})
        self._judge_fn: JudgeFn = judge_fn  # ensure set

    # ---- Optuna API ----
    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        layer = trial.suggest_categorical("layer", self.candidate_layers)
        alpha = trial.suggest_float("alpha", self.alpha_lo, self.alpha_hi)
        return {"layer": str(layer), "alpha": float(alpha)}

    def _judge_pair(self, prompt: Prompt, base: str, steered: str) -> int:
        # Judge must be robust to noisy outputs; return 1 if steered wins, else 0.
        q = next((m["content"] for m in prompt if m.get("role") == "user"), "")
        try:
            win = int(bool(self._judge_fn(q, base, steered)))
        except Exception:
            # If judge fails, fall back to tie (count as 0)
            win = 0
        return win

    def evaluate(self, trial: optuna.Trial, params: dict[str, Any]) -> float:
        layer = params["layer"]
        alpha = float(params["alpha"])
        vec = self.vectors_by_layer[layer]

        # Sample a subset for this trial to keep latency predictable
        import random as _r
        prompts = _r.sample(self.val_prompts, min(self.sample_size, len(self.val_prompts)))

        wins = 0
        seen = 0
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]

            # BASELINE
            base_out = self.wm.generate(batch, use_steering=False, **self.gen_kwargs)

            # STEERED
            self.wm.set_steering_from_raw({layer: vec}, scale=alpha, normalize=self.normalize_vectors)
            try:
                steered_out = self.wm.generate(batch, use_steering=True, **self.gen_kwargs)
            finally:
                self.wm.clear_steering()

            # Judge pairwise
            for p, A, B in zip(batch, base_out, steered_out):
                # Randomize A/B to reduce position bias
                if random.random() < 0.5:
                    win = self._judge_pair(p, A, B)
                else:
                    win = 1 - self._judge_pair(p, B, A)
                wins += win
                seen += 1

            # Report intermediate mean win-rate for pruning
            trial.report(wins / max(seen, 1), step=seen)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return wins / max(seen, 1)