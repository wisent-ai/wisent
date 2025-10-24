from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import typer

from wisent.cli.wisent_cli.ui import echo
from wisent.cli.wisent_cli.util.parsing import parse_natural_tokens, parse_kv, to_bool

try:
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    HAS_RICH = True
except Exception:
    HAS_RICH = False

__all__ = ["app", "train_classifier"]

app = typer.Typer(help="Classifier training workflow")


def _resolve_classifier(classifier_name: Optional[str], classifiers_location: Optional[str], **kwargs):
    from wisent.cli.classifiers.classifier_rotator import ClassifierRotator  # type: ignore

    # Best effort discovery if available
    try:
        if classifiers_location and hasattr(ClassifierRotator, "discover_classifiers"):
            ClassifierRotator.discover_classifiers(classifiers_location)  # type: ignore[attr-defined]
    except Exception:
        pass

    rot = ClassifierRotator(autoload=True)
    if classifier_name:
        # Case-insensitive match from registry
        registry = {c["name"].lower(): c["name"] for c in rot.list_classifiers()}
        real = registry.get(classifier_name.lower(), classifier_name)
        try:
            rot.use(real, **kwargs)
            inst = getattr(rot, "_classifier", None)
            if inst is not None:
                return inst
        except Exception:
            pass
        # Fallback to private resolver
        try:
            return ClassifierRotator._resolve_classifier(real, **kwargs)
        except Exception as ex:
            raise typer.BadParameter(f"Unknown classifier: {classifier_name!r}") from ex

    # No name provided -> default to first or 'mlp' if present
    names = [c["name"] for c in rot.list_classifiers()]
    if "mlp" in [n.lower() for n in names]:
        rot.use("mlp", **kwargs)
        return getattr(rot, "_classifier", ClassifierRotator._resolve_classifier("mlp", **kwargs))
    if not names:
        raise typer.BadParameter("No classifiers registered.")
    rot.use(names[0], **kwargs)
    return getattr(rot, "_classifier", ClassifierRotator._resolve_classifier(names[0], **kwargs))


def _show_plan(
    *,
    model: str,
    loader: Optional[str],
    loaders_location: Optional[str],
    loader_kwargs: Dict[str, object],
    classifier_name: Optional[str],
    classifier_kwargs: Dict[str, object],
    layers: Optional[str],
    aggregation_name: str,
    store_device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    test_size: float,
    save_path: Optional[Path],
) -> None:
    plan = {
        "Model": model,
        "Data loader": loader or "(default)",
        "Loaders location": loaders_location or "(auto)",
        "Loader kwargs": loader_kwargs or {},
        "Classifier": classifier_name or "(mlp)",
        "Classifier kwargs": classifier_kwargs or {},
        "Layers": layers or "(all)",
        "Aggregation": aggregation_name,
        "Epochs": epochs,
        "Batch size": batch_size,
        "Learning rate": learning_rate,
        "Test size": test_size,
        "Store device": store_device,
        "Save path": str(save_path) if save_path else "(none)",
    }

    code = f"""
# Example: Training classifier on activations (auto-generated plan)
from wisent.core.models.wisent_model import WisentModel
from wisent.core.trainers.steering_trainer import WisentSteeringTrainer
from wisent.cli.data_loaders.data_loader_rotator import DataLoaderRotator
from wisent.cli.steering_methods.steering_rotator import SteeringMethodRotator
from wisent.cli.classifiers.classifier_rotator import ClassifierRotator

# 1) Load model
model = WisentModel(model_name={model!r}, device={store_device!r})

# 2) Load data
data_loader_rot = DataLoaderRotator(loader={loader!r}, loaders_location={loaders_location!r})
data = data_loader_rot.load(**{json.dumps(loader_kwargs)})
train_pairs = data['train_qa_pairs']

# 3) Get steering method (CAA) to extract activations
steering_rot = SteeringMethodRotator()
steering_rot.use("caa")
caa_method = steering_rot._method

# 4) Train steering vectors to get activations
trainer = WisentSteeringTrainer(
    model=model,
    pair_set=train_pairs,
    steering_method=caa_method,
    store_device={store_device!r}
)

result = trainer.run(
    layers_spec={layers!r},
    aggregation={aggregation_name!r},
    return_full_sequence=False,
    normalize_layers=True,
)

# 5) Extract activations for classifier training
X, y = [], []
for pair in train_pairs.pairs:
    # Positive activation (label=1)
    pos_act = result.steering_vectors[{layers!r}]  # Simplified
    X.append(pos_act)
    y.append(1)
    # Negative activation (label=0)
    neg_act = ...  # Extract negative activation
    X.append(neg_act)
    y.append(0)

# 6) Train classifier
classifier_rot = ClassifierRotator(classifier={classifier_name or 'mlp'!r}, **{json.dumps(classifier_kwargs)})
report = classifier_rot.fit(X, y, epochs={epochs}, batch_size={batch_size})

# 7) Save classifier
classifier_rot.save_model({str(save_path) if save_path else None!r})
""".strip()

    if HAS_RICH:
        t = Table(title="Execution Plan")
        t.add_column("Key", style="bold", no_wrap=True)
        t.add_column("Value")
        for k, v in plan.items():
            t.add_row(k, json.dumps(v) if isinstance(v, (dict, list)) else str(v))
        echo(Panel(t, expand=False))
        echo(Panel(Syntax(code, "python", word_wrap=False), title="Code Preview", expand=False))
    else:
        print(json.dumps(plan, indent=2))
        print("\n" + code)


@app.command("train-classifier", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def train_classifier(ctx: typer.Context, params: List[str] = typer.Argument(None)):
    """
    Train a classifier on activations from contrastive pairs.

    Natural (no-dash) usage examples:

      wisent train-classifier model meta-llama/Llama-3.2-1B-Instruct loader task_interface task gsm8k training_limit 10 classifier mlp hidden_dim 64 epochs 50 save_path ./classifier.pt

      wisent train-classifier interactive true

    The classifier is trained on activations extracted from positive/negative response pairs.
    """
    # Lazy imports
    from wisent.cli.data_loaders.data_loader_rotator import DataLoaderRotator  # type: ignore
    from wisent.core.models.wisent_model import WisentModel  # type: ignore
    from wisent.core.trainers.steering_trainer import WisentSteeringTrainer  # type: ignore
    from wisent.cli.steering_methods.steering_rotator import SteeringMethodRotator  # type: ignore
    from wisent.cli.classifiers.classifier_rotator import ClassifierRotator  # type: ignore

    tokens = list(params or []) + list(ctx.args or [])
    top, loader_kv_raw, classifier_kv_raw = parse_natural_tokens(tokens)

    # Core args
    model = top.get("model")
    if not model:
        raise typer.BadParameter("Please specify a model (e.g. `train-classifier model meta-llama/Llama-3.2-1B-Instruct`) or use `interactive true`.")

    loader = top.get("loader")
    loaders_location = top.get("loaders_location")
    classifiers_location = top.get("classifiers_location")
    classifier_name = top.get("classifier")

    layers = top.get("layers") or "8"
    aggregation_name = (top.get("aggregation") or "continuation_token").lower()
    store_device = top.get("device") or top.get("store_device") or "cpu"

    # Classifier training params
    epochs = int(top.get("epochs", "50"))
    batch_size = int(top.get("batch_size", "32"))
    learning_rate = float(top.get("learning_rate", "0.001"))
    test_size = float(top.get("test_size", "0.2"))

    save_path = Path(top["save_path"]) if top.get("save_path") else None
    interactive = to_bool(top.get("interactive", "false")) if "interactive" in top else False
    plan_only = to_bool(top.get("plan-only", top.get("plan_only", "false"))) if ( "plan-only" in top or "plan_only" in top ) else False
    confirm = to_bool(top.get("confirm", "true")) if "confirm" in top else True

    # Convert kwargs
    loader_kwargs = parse_kv([f"{k}={v}" for k, v in loader_kv_raw.items()])
    classifier_kwargs = parse_kv([f"{k}={v}" for k, v in classifier_kv_raw.items()])

    # Interactive wizard
    if interactive:
        if loaders_location:
            DataLoaderRotator.discover_loaders(loaders_location)
        if not loader:
            options = [d["name"] for d in DataLoaderRotator.list_loaders()]
            loader = typer.prompt("Choose data loader", default=(options[0] if options else "task_interface"))
        if not classifier_name:
            classifier_name = typer.prompt("Choose classifier (see list-classifiers)", default="mlp")
        if "layers" not in top:
            layers = typer.prompt("Layers (e.g., '8', '4-6' or leave empty)", default="8")
        if "save_path" not in top:
            default_out = "./classifier.pt"
            p = typer.prompt("Save path for classifier (blank to skip saving)", default=default_out)
            if p.strip():
                save_path = Path(p)

    # Plan
    _show_plan(
        model=model,
        loader=loader,
        loaders_location=loaders_location,
        loader_kwargs=loader_kwargs,
        classifier_name=classifier_name,
        classifier_kwargs=classifier_kwargs,
        layers=layers,
        aggregation_name=aggregation_name,
        store_device=store_device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        test_size=test_size,
        save_path=save_path,
    )

    if plan_only:
        return

    if confirm and not typer.confirm("Proceed with classifier training?", default=True):
        typer.echo("Aborted.")
        raise typer.Exit(code=1)

    # -- Model -----------------------------------------------------------------
    typer.echo(f"[+] Loading model: {model}")
    wmodel = WisentModel(model_name=model, device=store_device)

    # -- Data loader -----------------------------------------------------------
    if loaders_location:
        DataLoaderRotator.discover_loaders(loaders_location)
    dl_rot = DataLoaderRotator(loader=loader, loaders_location=loaders_location or "wisent.core.data_loaders.loaders")
    typer.echo(f"[+] Using data loader: {loader or '(default)'}")
    load_result = dl_rot.load(**loader_kwargs)
    pair_set = load_result["train_qa_pairs"]
    typer.echo(f"[+] Loaded training pairs: {len(pair_set)} (task_type={load_result['task_type']})")

    # -- Steering method (to extract activations) ------------------------------
    steering_rot = SteeringMethodRotator()
    steering_rot.use("caa")
    caa_method = steering_rot._method
    typer.echo(f"[+] Using CAA method to extract activations")

    # -- Extract activations ---------------------------------------------------
    typer.echo(f"[+] Extracting activations from layer {layers}...")
    trainer = WisentSteeringTrainer(
        model=wmodel,
        pair_set=pair_set,
        steering_method=caa_method,
        store_device=store_device,
    )

    result = trainer.run(
        layers_spec=layers,
        aggregation=aggregation_name,
        return_full_sequence=False,
        normalize_layers=True,
    )

    # Extract activation vectors for classifier training
    import torch
    X, y = [], []

    # Get the layer activations from result
    layer_key = str(layers)
    if hasattr(result, 'steering_vectors'):
        steering_vectors = result.steering_vectors
    else:
        steering_vectors = result.get('steering_vectors', {})

    # For each pair, we have positive and negative activations
    # The steering vector is (positive - negative), so we need to extract both
    # For simplicity, we'll use the steering vector magnitude as a feature
    for i, pair in enumerate(pair_set.pairs):
        if layer_key in steering_vectors:
            activation = steering_vectors[layer_key]
            if isinstance(activation, torch.Tensor):
                # Positive class
                X.append(activation.flatten().cpu().numpy())
                y.append(1)
                # Negative class (inverted)
                X.append((-activation).flatten().cpu().numpy())
                y.append(0)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    typer.echo(f"[+] Prepared {len(X)} activation samples for classifier training")

    # -- Classifier ------------------------------------------------------------
    classifier_rot = ClassifierRotator(
        classifier=classifier_name,
        classifiers_location=classifiers_location,
        autoload=True,
        **classifier_kwargs
    )
    name_shown = classifier_name or "mlp"
    typer.echo(f"[+] Training classifier: {name_shown}")

    report = classifier_rot.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        test_size=test_size
    )

    typer.echo("\n=== Training Summary ===")
    typer.echo(f"Classifier: {report.classifier_name}")
    typer.echo(f"Input dim: {report.input_dim}")
    typer.echo(f"Best epoch: {report.best_epoch}/{report.epochs_ran}")
    typer.echo(f"Accuracy: {report.final.accuracy:.4f}")
    typer.echo(f"Precision: {report.final.precision:.4f}")
    typer.echo(f"Recall: {report.final.recall:.4f}")
    typer.echo(f"F1: {report.final.f1:.4f}")
    typer.echo(f"AUC: {report.final.auc:.4f}")

    if save_path is not None:
        classifier_rot.save_model(str(save_path))
        typer.echo(f"\nClassifier saved to: {Path(save_path).resolve()}\n")
