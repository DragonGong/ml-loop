# Script Layout

Scripts are grouped by workflow:

- `appworld/`: reusable AppWorld inference, metric parsing, data install, query
  visualization, episode visualization, and task-id utilities.
- `loop7b/`: Qwen2.5-7B LOOP experiment helpers for environment setup, split
  preparation, debug/stage2 training, checkpoint sync, checkpoint evaluation,
  episode summary, behavior analysis, and dev-eval watching.
- `diagnostics/`: runtime and distributed health checks.

Common entry points:

```bash
python -m scripts.appworld.run_inference
python -m scripts.appworld.eval_parse_and_log
python -m scripts.loop7b.summarize_appworld_episodes
python -m scripts.loop7b.analyze_appworld_behavior
python -m scripts.loop7b.eval_watch
python -m scripts.diagnostics.torch_dist_healthcheck
```
