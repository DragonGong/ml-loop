# AppWorld 教师轨迹收集阻塞事故与修复报告

## 摘要

2026-07-12，DeepSeek `deepseek-v4-flash-thinking` 教师轨迹收集出现两类中断：

1. 首次收集运行在交互式终端中，终端会话结束后 worker 被回收。
2. 改为后台运行后，worker 仍可能在单次 DeepSeek 非流式请求上长期阻塞；进程仍存活，但
   `generation_manifest.json` 与日志不再更新。

本次修复为每次 DeepSeek 请求增加绝对墙钟超时，明确选择 HTTP 代理并禁用 SDK 内部重试，超时
后重建 HTTP client；同时将收集脚本改为 worker 异常退出后从 manifest 自动重启。所有已提交的
轨迹均被保留。

## 影响范围

- 影响对象：`train_difficulty_1_2` 教师轨迹收集。
- 未受影响：已经成功写入 manifest 的原始轨迹与任务级统计。
- 丢弃范围：阻塞时正在执行、尚未写入 manifest 的单个 in-flight attempt；该任务会在下次
  重采样中重新执行。
- API key：未写入代码、日志、报告或数据集。

## 观察到的事实

### 首次中断

首次 worker 在约 01:07 停止，日志中没有 `Traceback`、费用上限、AppWorld 服务异常或 DeepSeek
HTTP 错误。该 worker 隶属于交互式终端会话，因此判定为会话回收，而不是任务逻辑失败。

### 长连接阻塞

后台 worker 后续在约 15:08 停止更新 manifest，但 Python 进程和 AppWorld server 仍存活。检查结果：

- AppWorld 本地健康检查正常。
- 生成器只有一个线程，处于网络轮询等待。
- 该进程保持到本地代理 `127.0.0.1:17890` 的已建立 TCP 连接。
- 本地 `mihomo` 代理仍在监听，经过该代理访问 DeepSeek endpoint 能快速收到认证失败响应，说明
  代理进程本身未整体宕机。

因此，直接证据表明阻塞点是 DeepSeek API 请求路径（客户端/代理连接），而非 AppWorld 环境。

## 根因分析

DeepSeek 官方说明，非流式请求在排队期间会持续发送空行以保持 TCP 连接；流式请求会发送
SSE keep-alive 注释。普通 HTTP `read timeout` 以“最后收到字节”的时间计时，收到这些 keep-alive
后会被刷新，不能提供整个请求的硬上限。

原实现只向 OpenAI SDK 传递了 `request_timeout_seconds=180`，且让 SDK 自动读取同时存在的
`HTTP(S)_PROXY` 与 `ALL_PROXY` 环境变量。该组合无法保证请求在长连接 keep-alive 场景下退出，
并可能在 HTTP/SOCKS 代理选择或连接复用上放大阻塞。

DeepSeek 的官方文档还说明，如果请求在 10 分钟内未开始推理，服务端会关闭连接；这不是客户端的
端到端 deadline，也不能防止代理或客户端在连接关闭/重试处理上继续等待。

参考：

- [DeepSeek Rate Limit & Isolation](https://api-docs.deepseek.com/quick_start/rate_limit/)
- [DeepSeek FAQ](https://api-docs.deepseek.com/faq/)
- [DeepSeek Service Status](https://status.deepseek.com/)

## 修复内容

### 1. 请求绝对墙钟 deadline

在 `phi_agents/api_eval/deepseek_appworld.py` 中新增：

- `DeepSeekAbsoluteTimeout`；
- `absolute_deadline()`；
- `DeepSeekProfile.absolute_request_timeout_seconds`，默认 300 秒。

每个 `chat.completions.create()` 由该 deadline 包裹。即使持续收到 keep-alive，超过 300 秒仍会
抛出可恢复异常。超时后关闭旧 client、重建连接，并将当前 task 作为失败 attempt 原子保存，使后续
重采样继续执行。

### 2. 显式 HTTP client 与代理

DeepSeek client 现在：

- 用分项 `httpx.Timeout` 限制 connect/read/write/pool；
- 只显式采用 `HTTPS_PROXY`；
- 通过 `trust_env=False` 禁止 httpx 自动混用 `ALL_PROXY`；
- 设置 OpenAI SDK `max_retries=0`，避免 SDK 内部重试绕过项目的 deadline 与审计逻辑；
- 继续使用项目已有的、带退避的外层重试。

### 3. 可恢复后台运行

`scripts/sft/run_teacher_collection.sh` 改为 `run_with_restart()`：如果 worker 因未预期错误退出，
脚本等待 15 秒后从已有 manifest 重启。收集进程以 `nohup + setsid` 运行，避免再次依赖交互终端。

## 数据完整性验证

修复前检查 manifest 时：

- 已记录 218 个 jobs；
- manifest 中每个 `trajectory_path` 都存在；
- 没有已提交轨迹丢失。

修复并重启后，第一个新任务在约 51 秒内完成，manifest 增至 219 个 jobs；再次检查确认所有 219 个
manifest 条目均有对应轨迹文件。新轨迹的 `model_config` 中已记录
`absolute_request_timeout_seconds: 300.0`。

## 验证

执行以下检查并通过：

```text
ruff check phi_agents/api_eval/deepseek_appworld.py \
  scripts/appworld/run_deepseek_dev_eval.py \
  tests/test_appworld_sft_pipeline.py
python -m compileall -q phi_agents/api_eval/deepseek_appworld.py
bash -n scripts/sft/run_teacher_collection.sh
pytest -q tests/test_appworld_sft_pipeline.py
```

新增测试验证绝对 deadline 不会因持续活动而被重置；共 7 项相关测试通过。

## 后续监控与建议

- 每次检查进度时同时查看 manifest 修改时间；超过 10 分钟未更新应调查。
- worker 退出会自动恢复；若 manifest 长时间不更新但进程仍存活，300 秒 API deadline 应使当前任务
  自动失败并继续。若仍发生，优先收集代理和 httpx 层日志。
- 保持按任务原子提交：不要删除或覆盖已有 `raw/<task_id>/attempt-*/trajectory.json`。
- 若出现 429/500/503，继续使用退避重试；DeepSeek 官方建议对 500/503 稍后重试、对 429 降速。
