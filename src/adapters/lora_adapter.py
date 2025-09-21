import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math


class LoRAAdapter(nn.Module):
    """LoRA adapter for efficient task adaptation.

    Implements Low-Rank Adaptation for parameter-efficient fine-tuning,
    as used in Jina-Embeddings-v3 for task-specific specialization.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 16,
            alpha: float = 32.0,
            dropout: float = 0.1,
            fan_in_fan_out: bool = False
    ):
        """Initialize LoRA adapter.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of adaptation matrices
            alpha: Scaling parameter
            dropout: Dropout probability
            fan_in_fan_out: Set for Conv1D layers
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.fan_in_fan_out = fan_in_fan_out

        # Create low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize weights
        self.reset_parameters()

        # Enable/disable adapter
        self.enabled = True

    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation.

        Args:
            x: Input tensor
            base_output: Output from base model

        Returns:
            Adapted output
        """
        if not self.enabled:
            return base_output

        # Apply LoRA
        if self.fan_in_fan_out:
            # For Conv1D layers
            lora_output = (
                                  self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
                          ) * self.scaling
        else:
            # Standard linear layers
            lora_output = (
                                  self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
                          ) * self.scaling

        return base_output + lora_output

    def merge_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge LoRA weights into base model.

        Args:
            base_weight: Base model weight matrix

        Returns:
            Merged weight matrix
        """
        if not self.enabled:
            return base_weight

        # Compute LoRA weight update
        lora_weight = (self.lora_B @ self.lora_A) * self.scaling

        # Merge with base weights
        merged = base_weight + lora_weight

        return merged


class MultiTaskLoRAAdapter(nn.Module):
    """Multi-task LoRA adapter supporting multiple tasks."""

    def __init__(
            self,
            base_model: nn.Module,
            task_configs: Dict[str, Dict],
            shared_rank: int = 8,
            task_specific_rank: int = 8
    ):
        """Initialize multi-task LoRA adapter.

        Args:
            base_model: Base embedding model
            task_configs: Configuration for each task
            shared_rank: Rank for shared adaptation
            task_specific_rank: Rank for task-specific adaptation
        """
        super().__init__()

        self.base_model = base_model
        self.tasks = list(task_configs.keys())

        # Shared adapter
        self.shared_adapter = self._create_adapters_for_model(
            base_model, shared_rank, prefix='shared'
        )

        # Task-specific adapters
        self.task_adapters = nn.ModuleDict()
        for task in self.tasks:
            self.task_adapters[task] = self._create_adapters_for_model(
                base_model, task_specific_rank, prefix=task
            )

        # Task weights for mixing
        self.task_weights = nn.ParameterDict({
            task: nn.Parameter(torch.ones(1))
            for task in self.tasks
        })

    def _create_adapters_for_model(
            self,
            model: nn.Module,
            rank: int,
            prefix: str
    ) -> nn.ModuleDict:
        """Create LoRA adapters for all linear layers in model."""
        adapters = nn.ModuleDict()

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                adapter_name = f"{prefix}_{name.replace('.', '_')}"
                adapters[adapter_name] = LoRAAdapter(
                    module.in_features,
                    module.out_features,
                    rank=rank
                )

        return adapters

    def forward(
            self,
            inputs: torch.Tensor,
            task: str,
            **kwargs
    ) -> torch.Tensor:
        """Forward pass with task-specific adaptation.

        Args:
            inputs: Input tensor
            task: Task name
            **kwargs: Additional arguments for base model

        Returns:
            Task-adapted embeddings
        """
        # Get base embeddings
        with torch.no_grad():
            base_output = self.base_model(inputs, **kwargs)

        # Apply shared adaptation
        adapted_output = base_output.clone()
        for name, adapter in self.shared_adapter.items():
            if adapter.enabled:
                layer_name = name.replace('shared_', '').replace('_', '.')
                # Get corresponding layer output and apply adapter
                # This is simplified - in practice would hook into model
                adapted_output = adapter(inputs, adapted_output)

        # Apply task-specific adaptation
        if task in self.task_adapters:
            task_weight = torch.sigmoid(self.task_weights[task])
            task_adapter = self.task_adapters[task]

            for name, adapter in task_adapter.items():
                if adapter.enabled:
                    # Apply task-specific adaptation with weighting
                    task_adapted = adapter(inputs, base_output)
                    adapted_output = (
                            task_weight * task_adapted +
                            (1 - task_weight) * adapted_output
                    )

        return adapted_output

    def set_task(self, task: str):
        """Set active task for inference."""
        # Disable all task adapters
        for t in self.tasks:
            for adapter in self.task_adapters[t].values():
                adapter.enabled = False

        # Enable selected task adapter
        if task in self.task_adapters:
            for adapter in self.task_adapters[task].values():
                adapter.enabled = True