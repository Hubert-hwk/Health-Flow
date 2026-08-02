"""Optional local data-generation entry point.

The old version embedded a provider credential.  This script deliberately
loads credentials from the environment and does not ship or generate data in
the public repository.
"""

from __future__ import annotations

import os


def main() -> int:
    if not os.getenv("MINIMAX_API_KEY"):
        raise SystemExit("请先在 .env 中配置 MINIMAX_API_KEY；不要把密钥写入脚本。")
    raise SystemExit(
        "数据生成脚本已改为安全占位入口。请在本地按授权数据源实现生成流程，"
        "不要将患者数据或供应商密钥提交到仓库。"
    )


if __name__ == "__main__":
    main()
