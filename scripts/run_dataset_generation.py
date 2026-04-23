#!/usr/bin/env python3
"""运行 SFT 数据集生成脚本。

用法：
    python scripts/run_dataset_generation.py

环境变量：
    MINIMAX_API_KEY: MiniMax API Key（必填）
"""

import os
import sys
import argparse

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.service.data_augment import DataAugmentationPipeline, AugmentConfig


def main():
    parser = argparse.ArgumentParser(description="生成 HealthFlow SFT 数据集")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("MINIMAX_API_KEY", ""),
        help="MiniMax API Key（也可通过环境变量 MINIMAX_API_KEY 设置）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sft/training_data.jsonl",
        help="输出文件路径",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=8000,
        help="目标数据集大小（默认 8000）",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("错误：必须设置 MINIMAX_API_KEY")
        print("方式一：export MINIMAX_API_KEY=your_key")
        print("方式二：python scripts/run_dataset_generation.py --api-key your_key")
        sys.exit(1)

    # 设置 API Key 到环境变量（让 config.py 能读到）
    os.environ["MINIMAX_API_KEY"] = args.api_key

    print(f"开始生成 SFT 数据集，目标规模：{args.size} 条")
    print(f"输出路径：{args.output}")
    print("-" * 50)

    # 配置
    config = AugmentConfig(
        target_size=args.size,
        source="llm",  # 使用 LLM 扩展生成
        output_path=args.output,
    )

    # 创建 Pipeline
    pipeline = DataAugmentationPipeline(config)

    def progress_callback(progress: float, status: str):
        print(f"[{progress:.0%}] {status}")

    pipeline.set_progress_callback(progress_callback)

    # 运行生成
    pairs = pipeline.run()

    # 保存
    saved_path = pipeline.save()
    print("-" * 50)
    print(f"生成完成！共 {len(pairs)} 条数据")
    print(f"已保存到：{saved_path}")

    # 统计
    stats = pipeline.get_stats()
    print("\n数据集统计：")
    for category, count in stats.get("by_category", {}).items():
        print(f"  {category}: {count} 条")


if __name__ == "__main__":
    main()
