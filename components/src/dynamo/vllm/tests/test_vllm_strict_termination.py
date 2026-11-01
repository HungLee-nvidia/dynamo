# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import TokenizersBackend
from vllm.sampling_params import SamplingParams

from dynamo.vllm.generation_policy import PASSTHROUGH_GENERATION_POLICY
from dynamo.vllm.handlers import DecodeWorkerHandler
from dynamo.vllm.strict_rl_controller import StrictRLWeightTransferController
from dynamo.vllm.strict_termination import StrictRLTermination

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.timeout(20),
]


@pytest.fixture
def engine():
    raw = Tokenizer(models.BPE({"a": 0, "b": 1, "!": 2, "<eos>": 3}, []))
    raw.decoder = decoders.Fuse()
    tokenizer = TokenizersBackend(tokenizer_object=raw, eos_token="<eos>")
    renderer = SimpleNamespace(tokenizer=tokenizer, get_eos_token_id=lambda: 3)
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=8),
        input_processor=SimpleNamespace(
            renderer=renderer,
            tokenizer=tokenizer,
            generation_config_fields={"eos_token_id": [3, 4]},
        ),
    )


def completion(ids=(), *, finish=None, reason=None, text=""):
    return SimpleNamespace(
        index=0,
        token_ids=list(ids),
        text=text,
        finish_reason=finish,
        stop_reason=reason,
    )


def validator(engine, **kwargs):
    return StrictRLTermination.from_engine(
        engine, SamplingParams(max_tokens=4, **kwargs), [0, 1], "strict-test"
    )


@pytest.mark.parametrize("prompt_size,count", [(2, 4), (6, 2)])
def test_length_requires_request_or_actual_context_boundary(engine, prompt_size, count):
    state = StrictRLTermination.from_engine(
        engine, SamplingParams(max_tokens=4), [0] * prompt_size, "length"
    )
    state = state.observe(completion([0] * count, finish="length"))
    assert state.finished


@pytest.mark.parametrize("prompt_size,count", [(2, 0), (2, 2), (2, 5), (7, 2)])
def test_length_rejects_short_or_overlong_output(engine, prompt_size, count):
    state = StrictRLTermination.from_engine(
        engine, SamplingParams(max_tokens=4), [0] * prompt_size, "length"
    )
    with pytest.raises(ValueError, match="strict RL"):
        state.observe(completion([0] * count, finish="length"))


def test_stream_counts_all_deltas_and_accepts_empty_terminal_delta(engine):
    initial = validator(engine)
    state = initial.observe(completion([0, 0]))
    state = state.observe(completion([0, 0]))
    state = state.observe(completion(finish="length"))
    assert state.finished and initial.token_ids == ()
    with pytest.raises(ValueError, match="strict RL"):
        state.observe(completion([0]))


@pytest.mark.parametrize(
    "ids,reason", [([3], None), ([4], 4), ([2], 2), ([0, 3], None)]
)
def test_stop_accepts_actual_primary_extra_eos_and_explicit_tokens(engine, ids, reason):
    params = SamplingParams(max_tokens=4, stop_token_ids=[2])
    state = StrictRLTermination.from_engine(engine, params, [0, 1], "token")
    assert state.observe(completion(ids, finish="stop", reason=reason)).finished
    assert params.eos_token_id is None and params.stop_token_ids == [2]


@pytest.mark.parametrize(
    "ids,reason",
    [([], None), ([0], None), ([0], 2), ([2], 99), ([3], True), ([3], "unknown")],
)
def test_stop_rejects_missing_or_inconsistent_proof(engine, ids, reason):
    state = validator(engine, stop_token_ids=[2])
    with pytest.raises(ValueError, match="strict RL"):
        state.observe(completion(ids, finish="stop", reason=reason))


def test_ignore_eos_does_not_disable_explicit_user_stops(engine):
    state = validator(engine, ignore_eos=True, stop_token_ids=[2])
    assert state.observe(completion([2], finish="stop", reason=2)).finished
    for token in (3, 4):
        with pytest.raises(ValueError, match="strict RL"):
            state.observe(completion([token], finish="stop"))


@pytest.mark.parametrize(
    "ids,reason,text", [([0, 3], None, ""), ([0, 2], 2, ""), ([0, 2], "!", "a!")]
)
def test_stop_rejects_count_equal_to_min_tokens(engine, ids, reason, text):
    state = validator(
        engine,
        min_tokens=2,
        stop_token_ids=[2],
        stop=["!"],
        include_stop_str_in_output=True,
    )
    with pytest.raises(ValueError, match="strict RL"):
        state.observe(completion(ids, finish="stop", reason=reason, text=text))


@pytest.mark.parametrize(
    "ids,reason,text",
    [([0, 0, 3], None, ""), ([0, 0, 2], 2, ""), ([0, 1, 2], "!", "ab!")],
)
def test_stop_accepts_first_token_after_min_tokens(engine, ids, reason, text):
    state = validator(
        engine,
        min_tokens=2,
        stop_token_ids=[2],
        stop=["!"],
        include_stop_str_in_output=True,
    )
    assert state.observe(
        completion(ids, finish="stop", reason=reason, text=text)
    ).finished


def test_installed_vllm_unmasks_stops_only_after_existing_min_tokens(engine):
    builtin = pytest.importorskip("vllm.v1.sample.logits_processor.builtin")
    params = validator(engine, min_tokens=2, stop_token_ids=[2]).sampling_params
    masked = builtin.MinTokensLogitsProcessor.add_request(params, None, [0])
    assert masked is not None and masked[2] == {2, 3, 4}
    assert builtin.MinTokensLogitsProcessor.add_request(params, None, [0, 0]) is None


@pytest.mark.parametrize("include,text", [(True, "ab!"), (False, "")])
def test_string_stop_replays_actual_tokens_with_included_or_empty_excluded_text(
    engine, include, text
):
    state = validator(engine, stop=["ab!"], include_stop_str_in_output=include)
    state = state.observe(completion([0]))
    state = state.observe(completion([1, 2], finish="stop", reason="ab!", text=text))
    assert state.finished


def test_excluded_string_stop_can_have_nonempty_visible_prefix(engine):
    state = validator(engine, stop=["b!"], include_stop_str_in_output=False)
    assert state.observe(
        completion([0, 1, 2], finish="stop", reason="b!", text="a")
    ).finished


@pytest.mark.parametrize(
    "ids,reason,text",
    [
        ([0, 1], "ab!", ""),
        ([0, 1, 2], "b!", ""),
        ([0, 1, 2], "ab!", "wrong"),
        ([], "ab!", ""),
    ],
)
def test_string_stop_rejects_forged_or_inconsistent_evidence(engine, ids, reason, text):
    state = validator(engine, stop=["ab!"], include_stop_str_in_output=False)
    with pytest.raises(ValueError, match="strict RL"):
        state.observe(completion(ids, finish="stop", reason=reason, text=text))


@pytest.mark.parametrize("include", [True, False])
def test_string_stop_can_end_inside_a_token(engine, include):
    raw = Tokenizer(models.BPE({"ab!tail": 0, "b": 1, "!": 2, "<eos>": 3}, []))
    raw.decoder = decoders.Fuse()
    tokenizer = TokenizersBackend(tokenizer_object=raw, eos_token="<eos>")
    engine.input_processor.renderer.tokenizer = tokenizer
    engine.input_processor.tokenizer = tokenizer
    state = validator(engine, stop=["!"], include_stop_str_in_output=include)
    assert state.observe(
        completion([0], finish="stop", reason="!", text="ab!" if include else "ab")
    ).finished


def test_string_stop_requires_actual_tokenizer_metadata(engine):
    engine.input_processor.tokenizer = None
    with pytest.raises(ValueError, match="strict RL"):
        validator(engine, stop=["!"])


@pytest.mark.parametrize("limit", [None, 0, True])
def test_unknown_context_limit_fails_closed(engine, limit):
    engine.model_config.max_model_len = limit
    with pytest.raises(ValueError, match="strict RL"):
        validator(engine)


def test_unknown_eos_metadata_fails_closed(engine):
    del engine.input_processor.renderer.get_eos_token_id
    with pytest.raises((AttributeError, ValueError)):
        validator(engine)


def test_unset_max_tokens_uses_only_the_actual_context_limit(engine):
    state = StrictRLTermination.from_engine(
        engine, SamplingParams(max_tokens=None), [0, 1], "default"
    )
    assert state.sampling_params.max_tokens == 6
    assert state.observe(completion([0] * 6, finish="length")).finished


@pytest.mark.parametrize("prompt", [[], [0] * 8])
def test_invalid_prompt_boundary_fails_closed(engine, prompt):
    with pytest.raises(ValueError, match="strict RL"):
        StrictRLTermination.from_engine(engine, SamplingParams(), prompt, "prompt")


def test_invalid_effective_max_tokens_fails_closed(engine):
    params = SamplingParams()
    params.max_tokens = True
    with pytest.raises(ValueError, match="strict RL"):
        StrictRLTermination.from_engine(engine, params, [0], "invalid")


@pytest.mark.parametrize(
    "output",
    [
        completion([True]),
        completion([-1]),
        completion([0], text=None),
        completion([0], reason=2),
        completion([0], finish="abort"),
        completion([0] * 4, finish="length", reason=2),
        completion([3], finish="stop"),
    ],
)
def test_malformed_delta_and_terminal_metadata_fail_closed(engine, output):
    state = validator(engine, min_tokens=2)
    with pytest.raises(ValueError, match="strict RL"):
        state.observe(output)


@pytest.fixture
def worker(engine):
    handler = object.__new__(DecodeWorkerHandler)
    handler.engine_client = engine
    handler.generation_policy = StrictRLWeightTransferController(
        engine, 8, asyncio.Lock(), MagicMock()
    )
    return handler


async def generate_chunks(worker, outputs):
    async def generate(*args, **kwargs):
        for output in outputs:
            yield SimpleNamespace(
                prompt_token_ids=[0, 1],
                outputs=[output],
                prompt_logprobs=None,
                num_cached_tokens=0,
                kv_transfer_params=None,
            )

    worker.engine_client.generate = generate
    return [
        chunk
        async for chunk in worker.generate_tokens(
            {"prompt_token_ids": [0, 1]},
            SamplingParams(max_tokens=2),
            "handler-test",
            report_kv_cache_hit=False,
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", "stop"])
async def test_handler_rejects_unproven_terminal_before_emitting_it(worker, finish):
    with pytest.raises(ValueError, match="strict RL"):
        await generate_chunks(worker, [completion([0], finish=finish)])


@pytest.mark.asyncio
async def test_handler_checks_cumulative_deltas_and_terminal_token(worker):
    chunks = await generate_chunks(
        worker, [completion([0]), completion([3], finish="stop")]
    )
    assert [chunk["token_ids"] for chunk in chunks] == [[0], [3]]
    assert chunks[-1]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_handler_rejects_stream_without_terminal_evidence(worker):
    with pytest.raises(ValueError, match="strict RL"):
        await generate_chunks(worker, [completion([0])])


@pytest.mark.asyncio
async def test_handler_rejects_empty_stream(worker):
    with pytest.raises(ValueError, match="strict RL"):
        await generate_chunks(worker, [])


@pytest.mark.asyncio
async def test_handler_rejects_engine_prompt_changes(worker):
    async def generate(*args, **kwargs):
        yield SimpleNamespace(
            prompt_token_ids=[0], outputs=[completion([3], finish="stop")]
        )

    worker.engine_client.generate = generate
    with pytest.raises(ValueError, match="admitted prompt"):
        _ = [
            chunk
            async for chunk in worker.generate_tokens(
                {"prompt_token_ids": [0, 1]},
                SamplingParams(max_tokens=2),
                "changed-prompt",
            )
        ]


@pytest.mark.asyncio
async def test_non_strict_handler_preserves_short_length_behavior(worker):
    worker.generation_policy = PASSTHROUGH_GENERATION_POLICY
    chunks = await generate_chunks(worker, [completion([0], finish="length")])
    assert chunks[-1]["finish_reason"] == "length"


@pytest.mark.parametrize(
    "entries",
    [
        None,
        [],
        [{0: SimpleNamespace(logprob=-0.1)}],
        [{3: SimpleNamespace(logprob=float("nan"))}],
    ],
)
def test_request_observer_rejects_invalid_sampled_logprobs(worker, entries):
    observer = worker.generation_policy.request_observer(
        SamplingParams(max_tokens=2, logprobs=0),
        {"prompt_token_ids": [0, 1]},
        "logprobs",
    )
    output = completion([3], finish="stop")
    output.logprobs = entries
    with pytest.raises(ValueError, match="logprob"):
        observer.observe(SimpleNamespace(prompt_token_ids=[0, 1], outputs=[output]))


@pytest.mark.asyncio
async def test_passthrough_policy_has_no_prompt_or_output_requirements():
    policy = PASSTHROUGH_GENERATION_POLICY
    await policy.admit_generation()
    observer = policy.request_observer(None, "text-only-prompt", "ordinary")
    observer.observe(None)
    observer.finish()
    assert observer is policy.request_observer(None, None, "another-request")


@pytest.mark.parametrize("prompt", [None, "text", {"prompt": "text"}])
def test_strict_request_observer_requires_native_token_prompt(worker, prompt):
    with pytest.raises(ValueError, match="token prompt"):
        worker.generation_policy.request_observer(
            SamplingParams(max_tokens=2), prompt, "invalid-prompt"
        )
