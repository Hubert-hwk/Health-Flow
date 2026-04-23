"""Training related schemas."""

from typing import Optional, List
from pydantic import BaseModel, Field


class DataAugmentRequest(BaseModel):
    """Data augmentation request."""

    source: str = Field(..., description="数据来源，pmc/literature/template")
    target_size: int = Field(8000, description="目标数据集大小")
    categories: Optional[List[str]] = Field(None, description="类别过滤")


class DataAugmentResponse(BaseModel):
    """Data augmentation response."""

    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    progress: float = Field(0, ge=0, le=1)
    output_path: Optional[str] = Field(None, description="输出路径")


class FinetuneRequest(BaseModel):
    """Fine-tuning request."""

    model_name: str = Field(..., description="基础模型名")
    dataset_path: str = Field(..., description="数据集路径")
    output_dir: str = Field(..., description="输出目录")
    method: str = Field("qlora", description="微调方法，qlora/lora/sft")
    lora_r: int = Field(64, description="LoRA rank")
    lora_alpha: int = Field(16, description="LoRA alpha")
    learning_rate: float = Field(2e-4)
    num_epochs: int = Field(3)
    batch_size: int = Field(4)


class FinetuneResponse(BaseModel):
    """Fine-tuning response."""

    task_id: str
    status: str
    progress: float
    model_path: Optional[str] = None


class DPORequest(BaseModel):
    """DPO training request."""

    model_name: str = Field(..., description="参考模型名")
    dataset_path: str = Field(..., description="偏好数据集路径")
    output_dir: str = Field(..., description="输出目录")
    beta: float = Field(0.1, description="DPO温度参数")
    num_epochs: int = Field(3)
    batch_size: int = Field(4)


class DPOResponse(BaseModel):
    """DPO training response."""

    task_id: str
    status: str
    progress: float
    model_path: Optional[str] = None
