from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DPOTrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path: str = "data/dpo/preferences.jsonl"
    output_dir: str = "outputs/dpo"
    learning_rate: float = 5e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 1.0
    beta: float = 0.1
    max_length: int = 1024
    max_prompt_length: int = 512
    use_lora: bool = True
    dry_run: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

