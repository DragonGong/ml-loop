from scripts.loop7b.recompute_offline_process_advantage import (
    ApiCall,
    Trace,
    Turn,
    parse_turns,
    score_trace_group,
)


def _turn(
    index: int,
    endpoint: str,
    *,
    fingerprint: str | None = None,
    failed: bool = False,
) -> Turn:
    return Turn(
        index=index,
        assistant_text="",
        observation="Execution failed." if failed else "Execution successful.",
        code="",
        calls=[ApiCall(endpoint=endpoint, fingerprint=fingerprint or endpoint)],
        execution_failed=failed,
    )


def _trace(
    label: str,
    ret: float,
    turns: list[Turn],
    *,
    success: bool = False,
    failures: tuple[str, ...] = (),
) -> Trace:
    return Trace(
        label=label,
        iteration=1,
        scenario_idx=0,
        rollout_idx=0,
        task_id="task_1",
        difficulty=1,
        ret=ret,
        strict_success=success,
        turns=turns,
        failed_requirements=failures,
    )


def test_parse_turns_accepts_redacted_actual_task_marker() -> None:
    messages = [
        {"role": "assistant", "content": "```python\napis.spotify.login()\n```"},
        {
            "role": "user",
            "content": (
                "Using these APIs, now generate code to solve the [REDACTED] task: "
                "do the real work"
            ),
        },
        {
            "role": "assistant",
            "content": "```python\napis.spotify.search_artists()\n```",
        },
        {"role": "user", "content": "Execution successful."},
    ]

    turns = parse_turns(messages)

    assert len(turns) == 1
    assert turns[0].endpoint_names == ("spotify.search_artists",)


def test_success_action_is_higher_at_first_business_divergence() -> None:
    success = _trace(
        "success",
        1.0,
        [_turn(1, "spotify.login"), _turn(2, "spotify.search_artists")],
        success=True,
    )
    failure = _trace(
        "failure",
        0.2,
        [_turn(1, "spotify.login"), _turn(2, "spotify.show_artist_following")],
    )

    scores, forks, _ = score_trace_group([success, failure])

    assert len(forks) == 1
    assert scores[("success", 2)].new_advantage > scores[("failure", 2)].new_advantage
    assert forks[0]["advantage_margin"] > 0
    assert forks[0]["attribution_eligible"] is True


def test_failed_turn_on_eventual_success_is_not_labeled_success_action() -> None:
    success = _trace(
        "success",
        1.0,
        [
            _turn(1, "spotify.login"),
            _turn(2, "spotify.search_artists", failed=True),
            _turn(3, "spotify.search_artists"),
        ],
        success=True,
    )
    failure = _trace(
        "failure",
        0.2,
        [_turn(1, "spotify.login"), _turn(2, "spotify.show_artist_following")],
    )

    _, forks, _ = score_trace_group([success, failure])

    assert forks[0]["attribution_eligible"] is False
    assert "higher_action_execution_failed" in forks[0]["attribution_exclusion_reasons"]


def test_repeated_api_tail_is_forced_negative() -> None:
    looping = _trace(
        "looping",
        0.8,
        [
            _turn(1, "spotify.login"),
            _turn(2, "spotify.show_liked_songs", fingerprint="same-query"),
            _turn(3, "spotify.show_liked_songs", fingerprint="same-query"),
            _turn(4, "spotify.show_liked_songs", fingerprint="same-query"),
        ],
    )
    peer = _trace("peer", 0.0, [_turn(1, "spotify.login")])

    scores, _, _ = score_trace_group([looping, peer])

    assert scores[("looping", 3)].repeat_ordinal > 0
    assert scores[("looping", 3)].new_advantage < 0
    assert scores[("looping", 4)].new_advantage < scores[("looping", 3)].new_advantage


def test_recoverable_error_penalizes_only_error_and_rewards_correction() -> None:
    recovered = _trace(
        "recovered",
        1.0,
        [
            _turn(1, "spotify.login"),
            _turn(2, "spotify.like_song", fingerprint="same-call", failed=True),
            _turn(3, "spotify.like_song", fingerprint="same-call"),
        ],
        success=True,
    )
    peer = _trace("peer", 0.2, [_turn(1, "spotify.login")])

    scores, _, recoveries = score_trace_group([recovered, peer])

    assert scores[("recovered", 2)].new_advantage < 0
    assert scores[("recovered", 3)].new_advantage > 0
    assert scores[("recovered", 3)].repeat_ordinal > 0
    assert scores[("recovered", 3)].repeat_component == 0
    assert recoveries[0]["locality_pass"] is True


def test_d3_wrong_amount_and_receiver_mutations_cannot_be_positive() -> None:
    trace = _trace(
        "ckpt-9",
        0.8,
        [
            _turn(1, "venmo.create_transaction"),
            _turn(2, "phone.send_text_message"),
        ],
        failures=(
            "assert the added transaction has amount private_data.grocery_cost.",
            "assert the added global_text_message's receiver_id is to "
            "private_data.friend_phone_user_id.",
        ),
    )

    scores, _, _ = score_trace_group([trace])

    assert scores[("ckpt-9", 1)].grounding_failures
    assert scores[("ckpt-9", 2)].grounding_failures
    assert scores[("ckpt-9", 1)].new_advantage < 0
    assert scores[("ckpt-9", 2)].new_advantage < 0
