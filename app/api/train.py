"""Training API endpoints.

提供数据增强、微调训练、DPO训练等接口。
这些接口主要用于触发训练流程，实际训练在后台异步执行。
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
import uuid
from datetime import datetime

from app.schema.train import (
    DataAugmentRequest,
    DataAugmentResponse,
    FinetuneRequest,
    FinetuneResponse,
    DPORequest,
    DPOResponse,
)


router = APIRouter()


# 训练任务状态存储（生产环境应使用Redis等）
_training_tasks = {}


def update_task_status(task_id: str, status: str, progress: float = 0.0, **kwargs):
    """更新训练任务状态。"""
    if task_id in _training_tasks:
        _training_tasks[task_id].update({
            "status": status,
            "progress": progress,
            "updated_at": datetime.now().isoformat(),
            **kwargs
        })
    else:
        _training_tasks[task_id] = {
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **kwargs
        }


def run_data_augmentation(task_id: str, request: DataAugmentRequest):
    """
    执行数据增强（后台任务）。

    Args:
        task_id: 任务ID
        request: 增强请求参数
    """
    from app.service.data_augment import DataAugmentationPipeline, AugmentConfig

    update_task_status(task_id, "STARTING", 0.0)

    try:
        update_task_status(task_id, "DOWNLOADING", 0.1)

        # 创建数据增强Pipeline
        config = AugmentConfig(
            target_size=request.target_size,
            source=request.source,
            categories=request.categories or [
                "体检报告解读",
                "指标异常问询",
                "科室分诊建议",
                "医疗安全问答"
            ]
        )
        pipeline = DataAugmentationPipeline(config)

        update_task_status(task_id, "PROCESSING", 0.3)

        # 运行增强流程
        pairs = pipeline.run()

        update_task_status(task_id, "GENERATING", 0.6)

        # 保存数据集
        output_path = f"./data/augmented_{task_id}.json"
        pipeline.save(output_path)

        # 获取统计信息
        stats = pipeline.get_stats()

        update_task_status(
            task_id, "COMPLETED", 1.0,
            output_path=output_path,
            stats=stats
        )

    except Exception as e:
        update_task_status(task_id, "FAILED", 0.0, error=str(e))


def run_finetune(task_id: str, request: FinetuneRequest):
    """
    执行模型微调（后台任务）。

    Args:
        task_id: 任务ID
        request: 微调请求参数
    """
    from app.service.vlm_tuner import VLMTuner, VLMTunerConfig

    update_task_status(task_id, "STARTING", 0.0)

    try:
        update_task_status(task_id, "PREPARING", 0.1)

        # 创建微调配置
        config = VLMTunerConfig(
            base_model=request.model_name,
            dataset_path=request.dataset_path,
            output_dir=request.output_dir,
            num_train_epochs=request.num_epochs,
            per_device_train_batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            use_qlora=(request.method == "qlora"),
            lora_rank=request.lora_r,
            lora_alpha=request.lora_alpha,
        )

        # 创建微调器
        tuner = VLMTuner(config)

        update_task_status(task_id, "TRAINING", 0.3)

        # 执行训练
        stats = tuner.train()

        update_task_status(task_id, "CHECKPOINTING", 0.8)

        # 保存模型
        model_path = tuner.save_model()

        update_task_status(
            task_id, "COMPLETED", 1.0,
            model_path=str(model_path),
            stats={
                "total_samples": stats.total_samples,
                "trainable_params": stats.trainable_parameters,
                "total_params": stats.total_parameters
            }
        )

    except Exception as e:
        update_task_status(task_id, "FAILED", 0.0, error=str(e))


def run_dpo_training(task_id: str, request: DPORequest):
    """
    执行DPO训练（后台任务）。

    Args:
        task_id: 任务ID
        request: DPO请求参数
    """
    from app.service.safety_dpo import SafetyDPOTrainer, DPOConfig

    update_task_status(task_id, "STARTING", 0.0)

    try:
        update_task_status(task_id, "PREPARING", 0.1)

        # 创建DPO配置
        config = DPOConfig(
            base_model=request.model_name,
            ref_model=request.model_name,
            dataset_path=request.dataset_path,
            output_dir=request.output_dir,
            num_train_epochs=request.num_epochs,
            per_device_train_batch_size=request.batch_size,
            beta=request.beta,
        )

        # 创建DPO训练器
        trainer = SafetyDPOTrainer(config)

        update_task_status(task_id, "TRAINING", 0.3)

        # 执行训练
        stats = trainer.train()

        update_task_status(task_id, "VALIDATING", 0.8)

        # 保存模型
        model_path = trainer.save_model()

        update_task_status(
            task_id, "COMPLETED", 1.0,
            model_path=str(model_path),
            stats={
                "total_pairs": stats.total_pairs,
                "trainable_params": stats.trainable_parameters,
                "total_params": stats.total_parameters
            }
        )

    except Exception as e:
        update_task_status(task_id, "FAILED", 0.0, error=str(e))


@router.post("/augment", response_model=DataAugmentResponse)
async def trigger_data_augmentation(
    request: DataAugmentRequest,
    background_tasks: BackgroundTasks
):
    """
    触发数据增强任务。

    Args:
        request: 增强请求
        background_tasks: 后台任务

    Returns:
        任务信息
    """
    task_id = str(uuid.uuid4())

    # 创建任务记录
    update_task_status(task_id, "QUEUED", 0.0)

    # 添加后台任务
    background_tasks.add_task(run_data_augmentation, task_id, request)

    return DataAugmentResponse(
        task_id=task_id,
        status="QUEUED",
        progress=0.0,
        output_path=None
    )


@router.get("/augment/{task_id}", response_model=DataAugmentResponse)
async def get_augment_status(task_id: str):
    """
    获取数据增强任务状态。

    Args:
        task_id: 任务ID

    Returns:
        任务状态
    """
    if task_id not in _training_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = _training_tasks[task_id]

    return DataAugmentResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        output_path=task.get("output_path")
    )


@router.post("/finetune", response_model=FinetuneResponse)
async def trigger_finetune(
    request: FinetuneRequest,
    background_tasks: BackgroundTasks
):
    """
    触发模型微调任务。

    Args:
        request: 微调请求
        background_tasks: 后台任务

    Returns:
        任务信息
    """
    task_id = str(uuid.uuid4())

    # 创建任务记录
    update_task_status(task_id, "QUEUED", 0.0)

    # 添加后台任务
    background_tasks.add_task(run_finetune, task_id, request)

    return FinetuneResponse(
        task_id=task_id,
        status="QUEUED",
        progress=0.0,
        model_path=None
    )


@router.get("/finetune/{task_id}", response_model=FinetuneResponse)
async def get_finetune_status(task_id: str):
    """
    获取微调任务状态。

    Args:
        task_id: 任务ID

    Returns:
        任务状态
    """
    if task_id not in _training_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = _training_tasks[task_id]

    return FinetuneResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        model_path=task.get("model_path")
    )


@router.post("/dpo", response_model=DPOResponse)
async def trigger_dpo(
    request: DPORequest,
    background_tasks: BackgroundTasks
):
    """
    触发DPO训练任务。

    Args:
        request: DPO请求
        background_tasks: 后台任务

    Returns:
        任务信息
    """
    task_id = str(uuid.uuid4())

    # 创建任务记录
    update_task_status(task_id, "QUEUED", 0.0)

    # 添加后台任务
    background_tasks.add_task(run_dpo_training, task_id, request)

    return DPOResponse(
        task_id=task_id,
        status="QUEUED",
        progress=0.0,
        model_path=None
    )


@router.get("/dpo/{task_id}", response_model=DPOResponse)
async def get_dpo_status(task_id: str):
    """
    获取DPO任务状态。

    Args:
        task_id: 任务ID

    Returns:
        任务状态
    """
    if task_id not in _training_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = _training_tasks[task_id]

    return DPOResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        model_path=task.get("model_path")
    )


@router.get("/tasks")
async def list_tasks():
    """
    列出所有训练任务。

    Returns:
        任务列表
    """
    return {
        "tasks": list(_training_tasks.values()),
        "count": len(_training_tasks)
    }


@router.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """
    取消训练任务。

    Args:
        task_id: 任务ID

    Returns:
        取消结果
    """
    if task_id not in _training_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = _training_tasks[task_id]

    if task["status"] in ["COMPLETED", "FAILED"]:
        return {"message": "任务已完成或失败，无法取消", "task_id": task_id}

    update_task_status(task_id, "CANCELLED", task["progress"])

    return {"message": "任务已取消", "task_id": task_id}
