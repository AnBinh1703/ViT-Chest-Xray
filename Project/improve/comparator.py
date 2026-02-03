"""
Loss Function Comparison Utilities
==================================

Tools for comparing different loss functions and saving results.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from .utils import create_loss_function


class LossComparator:
    """
    Utility class for comparing different loss functions.

    Useful for experiments to find the best loss for your dataset.

    Example:
        >>> comparator = LossComparator(['focal', 'asymmetric', 'combined'])
        >>> results = comparator.compare(model, dataloader, device)
    """

    def __init__(
        self,
        loss_names: List[str],
        pos_weight: Optional[torch.Tensor] = None
    ):
        self.loss_functions = {
            name: create_loss_function(name, pos_weight=pos_weight)
            for name in loss_names
        }

    def compute_all(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all losses for given inputs and targets."""
        results = {}
        for name, loss_fn in self.loss_functions.items():
            with torch.no_grad():
                loss_value = loss_fn(inputs, targets)
                results[name] = loss_value.item()
        return results

    def summary(self) -> str:
        """Get summary of configured loss functions."""
        lines = ["Configured Loss Functions:"]
        for name, loss_fn in self.loss_functions.items():
            lines.append(f"  - {name}: {loss_fn}")
        return "\n".join(lines)


def save_loss_comparison_results(
    results: Dict[str, float],
    save_path: Union[str, Path],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save loss comparison results to JSON file.

    Args:
        results: Dictionary of loss names to values
        save_path: Path to save the JSON file
        metadata: Optional metadata (e.g., dataset info, hyperparameters)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'loss_values': results,
        'metadata': metadata or {}
    }

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {save_path}")