from omegaconf import OmegaConf

from phi_agents.rl.config import get_config


def _agent_config_for_params(params_name: str) -> dict:
    cfg = get_config(
        mode="train",
        overrides=[
            "+global@_global_=appworld",
            "rl/gpu_allocation=single_gpu",
            f"rl/params={params_name}",
            "llm=qwen_2_5_7b_lora16_train",
            "experiment_name=test_prompt_fairness",
        ],
    )
    return OmegaConf.to_container(
        cfg.rl.scenario_runner.appworld_config.agent,
        resolve=True,
    )


def test_grpo_config_uses_200_iterations() -> None:
    cfg = get_config(
        mode="train",
        overrides=[
            "+global@_global_=appworld",
            "rl/gpu_allocation=single_gpu",
            "rl/params=grpo",
            "llm=qwen_2_5_7b_lora16_train",
            "experiment_name=test_grpo_config",
        ],
    )

    assert cfg.rl.params.algorithm == "grpo"
    assert cfg.rl.params.total_iterations == 200
    assert cfg.rl.params.loss_type == "pg_per_token"
    assert cfg.rl.params.do_ppo_clipping is True
    assert cfg.rl.params.ppo_epsilon == 0.1


def test_loop_and_grpo_use_identical_appworld_react_agent_config() -> None:
    loop_agent = _agent_config_for_params("default")
    grpo_agent = _agent_config_for_params("grpo")

    assert grpo_agent == loop_agent
