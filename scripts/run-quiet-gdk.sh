#!/usr/bin/env bash
# Run a command and filter GDK 的 "Param ... not found" 噪音日志 (config_parser.cpp:27).
#
# 这条 E 级日志来自闭源 libgdk_core.so 内置参数表查不到 omnipicker idx31/71,
# 改任何磁盘配置都消不掉, 但完全不影响实际控制. 用本脚本包一层 stderr 过滤即可.
#
# 用法:
#   scripts/run-quiet-gdk.sh uv run python tests/g2/test_servo_joint_control.py
#   scripts/run-quiet-gdk.sh uv run lerobot-rollout-g2 --strategy.type=base_panic ...
#
# 也可以追加更多想过滤的 pattern, 用 EXTRA_FILTER 环境变量传入扩展正则 (egrep 语法):
#   EXTRA_FILTER='aorta]Connect server failed|tcp_subscriber\.cc' scripts/run-quiet-gdk.sh <cmd>
set -euo pipefail

PATTERN='config_parser\.cpp:.*\[GDK\]Param .* not found'
if [[ -n "${EXTRA_FILTER:-}" ]]; then
  PATTERN="${PATTERN}|${EXTRA_FILTER}"
fi

# 用 process substitution + grep --line-buffered, 保证子进程退出码能透传出来
exec "$@" 2> >(grep --line-buffered -Ev "${PATTERN}" >&2)

