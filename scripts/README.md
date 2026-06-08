# Script Layout

Scripts are grouped by workflow:

- `appworld/`: reusable AppWorld inference, metric parsing, data install, query
  visualization, episode visualization, and task-id utilities.
- `loop7b/`: Qwen2.5-7B LOOP experiment helpers for environment setup, small
  split preparation, debug training, small training, checkpoint evaluation, and
  episode summary.
- `diagnostics/`: runtime and distributed health checks.

Common entry points:

```bash
python -m scripts.appworld.run_inference
python -m scripts.appworld.eval_parse_and_log
python -m scripts.loop7b.summarize_appworld_episodes
python -m scripts.diagnostics.torch_dist_healthcheck
```
