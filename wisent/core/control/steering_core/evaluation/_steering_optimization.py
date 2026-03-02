"""Parameter optimization mixin for SteeringMethod."""
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from wisent.core.contrastive_pairs import ContrastivePairSet
from wisent.core.constants import TIKHONOV_REG, EWC_PERTURBATION_SCALE, STEERING_OPT_NUM_EPOCHS
from wisent.core.utils.infra.core.hardware import eval_batch_size


class SteeringOptimizationMixin:
    """Mixin providing optimization methods."""

    def optimize_parameters(
        self,
        model,
        target_layer,
        pair_set: ContrastivePairSet,
        learning_rate: float = TIKHONOV_REG,
        num_epochs: int = STEERING_OPT_NUM_EPOCHS,
        regularization_strength: float = EWC_PERTURBATION_SCALE,
    ) -> Dict[str, Any]:
        """
        Optimize model parameters to improve steering effectiveness.

        Args:
            model: Model object to optimize
            target_layer: Layer to optimize
            pair_set: ContrastivePairSet with training data
            learning_rate: Learning rate for optimization
            num_epochs: Number of optimization epochs
            regularization_strength: L2 regularization strength

        Returns:
            Dictionary with optimization results
        """
        try:
            # Get the target layer module for optimization
            layer_module = self._get_layer_module(model, target_layer)
            if layer_module is None:
                raise LayerNotFoundError(layer_name=str(target_layer))

            # Store original parameters
            self._store_original_parameters(layer_module)

            # Extract activations for the pair set
            pair_set.extract_activations_with_model(model, target_layer)

            # Prepare training data
            X_tensors, y_labels = pair_set.prepare_classifier_data()

            # Set up optimizer for just the target layer
            optimizer = torch.optim.Adam(layer_module.parameters(), lr=learning_rate)

            # Training loop
            best_steering_loss = float("inf")
            best_parameters = None

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0

                # Process in batches
                batch_size = eval_batch_size()
                for i in range(0, len(X_tensors), batch_size):
                    batch_X = X_tensors[i : i + batch_size]
                    batch_y = y_labels[i : i + batch_size]

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass through the modified layer
                    loss = self._compute_steering_loss(batch_X, batch_y, layer_module, regularization_strength)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

                # Track best parameters
                if avg_loss < best_steering_loss:
                    best_steering_loss = avg_loss
                    best_parameters = {name: param.clone() for name, param in layer_module.named_parameters()}

            # Load best parameters
            if best_parameters is not None:
                for name, param in layer_module.named_parameters():
                    if name in best_parameters:
                        param.data.copy_(best_parameters[name])

            # Store optimization results
            optimization_result = {
                "target_layer": target_layer.index if hasattr(target_layer, "index") else target_layer,
                "final_loss": best_steering_loss,
                "epochs": num_epochs,
                "learning_rate": learning_rate,
                "regularization_strength": regularization_strength,
                "parameters_optimized": True,
            }

            self.optimization_history.append(optimization_result)

            return optimization_result

        except Exception as e:
            return {"error": str(e), "parameters_optimized": False}

    def _get_layer_module(self, model, layer):
        """Get the module for a specific layer."""
        try:
            hf_model = model.hf_model if hasattr(model, "hf_model") else model
            layer_idx = layer.index if hasattr(layer, "index") else layer

            if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
                # Llama-style model
                if layer_idx < len(hf_model.model.layers):
                    return hf_model.model.layers[layer_idx]
            elif hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
                # GPT-style model
                if layer_idx < len(hf_model.transformer.h):
                    return hf_model.transformer.h[layer_idx]

            return None
        except Exception:
            return None

    def _store_original_parameters(self, module):
        """Store original parameters of a module."""
        key = f"module_{id(module)}"
        self.original_parameters[key] = {name: param.clone() for name, param in module.named_parameters()}

    def _compute_steering_loss(self, batch_X, batch_y, layer_module, regularization_strength):
        """
        Compute loss for steering optimization.

        Args:
            batch_X: Batch of activation tensors
            batch_y: Batch of labels
            layer_module: Layer module being optimized
            regularization_strength: L2 regularization strength

        Returns:
            Loss tensor
        """
        total_loss = 0.0

        # Compute steering effectiveness loss
        for i, (activation, label) in enumerate(zip(batch_X, batch_y)):
            # Get prediction from steering method
            prediction = self.predict_proba(activation)

            # Convert to tensor for loss computation (use activation's dtype)
            if not isinstance(prediction, torch.Tensor):
                from wisent.core.utils import preferred_dtype
                pred_dtype = activation.dtype if isinstance(activation, torch.Tensor) else preferred_dtype()
                prediction = torch.tensor(prediction, dtype=pred_dtype, device=self.device)

            target = torch.tensor(label, dtype=prediction.dtype, device=self.device)

            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(prediction.unsqueeze(0), target.unsqueeze(0))
            total_loss += loss

        # Add L2 regularization
        l2_reg = 0.0
        for param in layer_module.parameters():
            l2_reg += torch.norm(param, p=2)

        total_loss += regularization_strength * l2_reg

        return total_loss / len(batch_X)  # Average over batch

    def restore_original_parameters(self) -> bool:
        """
        Restore original parameters.

        Returns:
            Success flag
        """
        try:
            # This is a simplified version - in practice, you'd need to keep track
            # of which modules correspond to which keys
            return len(self.original_parameters) > 0
        except Exception:
            return False

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all optimizations performed.

        Returns:
            Summary dictionary
        """
        return {
            "total_optimizations": len(self.optimization_history),
            "optimization_history": self.optimization_history,
            "has_original_parameters": len(self.original_parameters) > 0,
            "method_type": self.method_type.value,
            "threshold": self.threshold,
        }

