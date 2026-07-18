from unittest.mock import MagicMock, call

from phi_agents.rl.llm import SeededTrainableLLM


def test_seeded_llm_derives_a_stable_seed_per_generation() -> None:
    llm = MagicMock()
    seeded = SeededTrainableLLM(llm, rollout_seed=20260713)

    seeded.generate([])
    seeded.generate([])

    assert llm.generate.call_args_list == [
        call([], seed=20260713),
        call([], seed=20260714),
    ]


def test_seeded_llm_preserves_an_explicit_request_seed() -> None:
    llm = MagicMock()
    seeded = SeededTrainableLLM(llm, rollout_seed=10)

    seeded.generate([], seed=99)
    seeded.generate([])

    assert llm.generate.call_args_list == [call([], seed=99), call([], seed=11)]


def test_seeded_llm_delegates_token_and_policy_methods() -> None:
    llm = MagicMock()
    llm.get_tokens.return_value = ([1], [True], [-0.5])
    llm.get_policy_token_info.return_value = "policy-info"
    seeded = SeededTrainableLLM(llm, rollout_seed=1)

    assert seeded.get_tokens([], is_output=True, log_probs=True) == (
        [1],
        [True],
        [-0.5],
    )
    assert seeded.get_policy_token_info([]) == "policy-info"
