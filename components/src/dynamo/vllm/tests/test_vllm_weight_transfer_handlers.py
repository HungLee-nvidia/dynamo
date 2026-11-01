# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.common.rl import RLRouteRegistry
from dynamo.vllm.args import Config, DynamoRuntimeConfig, cross_validate_config
from dynamo.vllm.backend_args import DynamoVllmArgGroup, DynamoVllmConfig
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.handlers import DecodeWorkerHandler
from dynamo.vllm.strict_rl_controller import StrictRLWeightTransferController
from dynamo.vllm.worker_factory import WorkerFactory

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.timeout(20),
]


@pytest.fixture
def handler():
    engine = SimpleNamespace(
        init_weight_transfer_engine=AsyncMock(),
        start_weight_update=AsyncMock(),
        update_weights=AsyncMock(),
        finish_weight_update=AsyncMock(),
        reset_prefix_cache=AsyncMock(return_value=True),
        resume_generation=AsyncMock(),
        pause_generation=AsyncMock(),
        collective_rpc=AsyncMock(return_value=[{"device_uuid": "GPU-test"}]),
        get_weight_version=AsyncMock(return_value="policy-1"),
        compute_weight_checksums=AsyncMock(),
    )
    return StrictRLWeightTransferController(
        engine, 128, asyncio.Lock(), MagicMock(side_effect=RuntimeError("dead"))
    )


async def begin(handler):
    assert (await handler.pause_generation({"mode": "abort"}))["status"] == "ok"
    assert (await handler.init_weight_transfer_engine({"init_info": {"packed": True}}))[
        "status"
    ] == "ok"
    assert (
        await handler.begin_weight_update(
            {"session_id": "round-1", "weight_version": "policy-1"}
        )
    )["status"] == "ok"


@pytest.mark.asyncio
async def test_handler_native_lifecycle_and_version_readback(handler):
    await begin(handler)
    status = await handler.get_weight_version({})
    assert status["pending_session_id"] == "round-1"
    assert status["model_max_len"] == 128
    assert status["device_uuids"] == ["GPU-test"]
    assert (await handler.start_weight_update({"session_id": "round-1", "seq": 0}))[
        "status"
    ] == "ok"
    assert (await handler.resume_generation({}))["status"] == "error"
    assert (
        await handler.update_weights(
            {"session_id": "round-1", "seq": 1, "update_info": {"names": ["w"]}}
        )
    )["status"] == "ok"
    assert handler.engine_client.update_weights.await_args.args[0].update_info == {
        "names": ["w"]
    }
    handler.engine_client.collective_rpc.assert_awaited_with("synchronize_device")
    assert (await handler.finish_weight_update({"session_id": "round-1", "seq": 2}))[
        "status"
    ] == "ok"
    handler.engine_client.finish_weight_update.assert_awaited_once_with("policy-1")
    assert (await handler.resume_generation({}))["status"] == "error"
    assert (await handler.flush_cache({}))["status"] == "ok"
    status = await handler.get_weight_version({})
    assert status["install_complete"] is True and status["session_id"] == "round-1"
    assert (await handler.resume_generation({}))["status"] == "ok"
    await handler.admit_generation()
    await handler.pause_generation({"mode": "wait"})
    with pytest.raises(ValueError, match="paused"):
        await handler.admit_generation()


@pytest.mark.parametrize(
    "route",
    [
        "set_weight_version",
        "update_weights_from_disk",
        "update_weights_from_distributed",
        "init_weights_update_group",
        "destroy_weights_update_group",
        "sleep",
        "wake_up",
        "scale_elastic_ep",
    ],
)
def test_strict_controller_omits_legacy_mutations(handler, route):
    assert route not in handler.routes()


@pytest.mark.asyncio
async def test_cancelled_start_latches_and_blocks_resume(handler):
    await begin(handler)
    handler.engine_client.start_weight_update.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await handler.start_weight_update({"session_id": "round-1", "seq": 0})
    assert (await handler.resume_generation({}))["status"] == "error"
    handler.engine_client.resume_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_strict_pause_rejects_keep_and_cache_mutation(handler):
    assert (await handler.pause_generation({}))["status"] == "error"
    assert (await handler.pause_generation({"mode": "abort", "clear_cache": True}))[
        "status"
    ] == "error"
    handler.engine_client.pause_generation.assert_not_awaited()
    assert (await handler.pause_generation({"mode": "abort"}))["status"] == "ok"


@pytest.mark.asyncio
async def test_resume_cannot_enter_between_start_and_finish(handler):
    await begin(handler)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held_start():
        entered.set()
        await release.wait()

    handler.engine_client.start_weight_update.side_effect = held_start
    task = asyncio.create_task(
        handler.start_weight_update({"session_id": "round-1", "seq": 0})
    )
    await entered.wait()
    resume = asyncio.create_task(handler.resume_generation({}))
    release.set()
    assert (await task)["status"] == "ok"
    assert (await resume)["status"] == "error"
    handler.engine_client.resume_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_generation_is_closed_before_install(handler):
    with pytest.raises(ValueError, match="installed"):
        await handler.admit_generation()


@pytest.mark.parametrize("enabled", [False, True])
def test_lifecycle_routes_are_registered_only_when_enabled(enabled):
    runtime = MagicMock()
    worker = MagicMock(spec=DecodeWorkerHandler)
    worker.config = SimpleNamespace(enable_rl=True, enable_rl_weight_transfer=enabled)
    worker.rl_route_registry = RLRouteRegistry(runtime)
    worker.model_max_len = 128
    worker.model_config = SimpleNamespace(is_moe=False, is_quantized=False)
    worker.use_vllm_tokenizer = False
    worker.model_config.dtype = torch.bfloat16
    worker._multimodal_request_processor = SimpleNamespace(enable_multimodal=False)
    worker.engine_client = MagicMock()
    worker._pause_lock = asyncio.Lock()
    WorkerFactory.register_engine_routes(None, runtime, MagicMock(), worker)
    paths = {item.args[0] for item in runtime.register_engine_route.call_args_list}
    assert ("update/init_weight_transfer_engine" in paths) is enabled
    assert (
        "update/begin_weight_update" in worker.rl_route_registry.describe()["routes"]
    ) is enabled
    legacy = {
        "control/sleep",
        "control/wake_up",
        "control/scale_elastic_ep",
        "control/ep_capacity",
        "set_weight_version",
        "update_weights_from_disk",
        "update_weights_from_distributed",
        "update_weights_from_tensor",
        "init_weights_update_group",
        "destroy_weights_update_group",
    }
    assert not paths.intersection(legacy) if enabled else legacy <= paths
    if enabled:
        expected = set(worker.generation_policy.routes()) | {
            "liveness_probe",
            "abort_request",
        }
        assert set(worker.rl_route_registry.describe()["routes"]) == expected


def test_strict_flag_defaults_off_and_parses_opt_in():
    parser = argparse.ArgumentParser()
    DynamoVllmArgGroup().add_arguments(parser)
    assert parser.parse_args([]).enable_rl_weight_transfer is False
    assert (
        parser.parse_args(["--enable-rl-weight-transfer"]).enable_rl_weight_transfer
        is True
    )


@pytest.mark.parametrize(
    "rl,secret,valid",
    [
        (False, "a" * 40, False),
        (True, None, False),
        (True, " ", False),
        (True, "short-token", False),
        (True, "a" * 32 + " ", False),
        (True, "a" * 32 + "\n", False),
        (True, "a" * 32 + "\x7f", False),
        (True, "a" * 32 + "é", False),
        (True, "a" * 40, True),
    ],
)
def test_strict_startup_requires_rl_and_authentication(monkeypatch, rl, secret, valid):
    if secret is None:
        monkeypatch.delenv("DYN_SYSTEM_ENGINE_AUTH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("DYN_SYSTEM_ENGINE_AUTH_TOKEN", secret)
    config = SimpleNamespace(enable_rl=rl, enable_rl_weight_transfer=True)
    with (
        patch.object(DynamoRuntimeConfig, "validate"),
        patch.object(DynamoVllmConfig, "validate"),
    ):
        if valid:
            Config.validate(config)
        else:
            with pytest.raises(ValueError):
                Config.validate(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 2),
        ("data_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("enable_lora", True),
        ("speculative_config", {}),
        ("weight_transfer_config", None),
        ("weight_transfer_config", SimpleNamespace(backend="unknown")),
        ("generation_config", "auto"),
        ("override_generation_config", {"temperature": 0.5}),
    ],
)
def test_strict_profile_rejects_unsupported_engine_configuration(field, value):
    config = SimpleNamespace(
        enable_rl_weight_transfer=True,
        disaggregation_mode=DisaggregationMode.AGGREGATED,
    )
    engine = SimpleNamespace(
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        enable_lora=False,
        speculative_config=None,
        weight_transfer_config=SimpleNamespace(backend="ipc"),
        generation_config="vllm",
        override_generation_config={},
    )
    setattr(engine, field, value)
    with pytest.raises(ValueError):
        cross_validate_config(config, engine)


@pytest.mark.asyncio
async def test_handler_rejects_false_cache_reset_and_native_version_mismatch(handler):
    await begin(handler)
    await handler.start_weight_update({"session_id": "round-1", "seq": 0})
    await handler.update_weights({"session_id": "round-1", "seq": 1, "update_info": {}})
    await handler.finish_weight_update({"session_id": "round-1", "seq": 2})
    handler.engine_client.get_weight_version.return_value = "wrong-policy"
    assert (await handler.get_weight_version({}))["status"] == "error"
    assert (await handler.resume_generation({}))["status"] == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pause_generation", "resume_generation"])
async def test_handler_pause_resume_engine_failure_latches(handler, phase):
    if phase == "resume_generation":
        await begin(handler)
        await handler.start_weight_update({"session_id": "round-1", "seq": 0})
        await handler.update_weights(
            {"session_id": "round-1", "seq": 1, "update_info": {}}
        )
        await handler.finish_weight_update({"session_id": "round-1", "seq": 2})
        await handler.flush_cache({})
    getattr(handler.engine_client, phase).side_effect = RuntimeError(
        "engine unavailable"
    )
    body = {"mode": "abort"} if phase == "pause_generation" else {}
    assert (await getattr(handler, phase)(body))["status"] == "error"
    assert handler.session.snapshot().failed is True


@pytest.mark.asyncio
async def test_status_exposes_only_explicitly_requested_readonly_checksums(handler):
    handler.engine_client.compute_weight_checksums.return_value = [
        {"dp0:pp0:pcp0:tp0:weight": "a" * 64}
    ]
    status = await handler.get_weight_checksums({})
    assert status["weight_checksums"] == {"dp0:pp0:pcp0:tp0:weight": "a" * 64}
    assert [
        call.args[0] for call in handler.engine_client.collective_rpc.await_args_list
    ] == ["get_weight_transfer_device_info"]
    handler.engine_client.compute_weight_checksums.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_status_rejects_checksum_option_before_rpc(handler):
    assert (await handler.get_weight_version({"include_weight_checksums": "true"}))[
        "status"
    ] == "error"
    handler.engine_client.collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_caches_immutable_identity_and_versions_protocol(handler):
    for _ in range(3):
        status = await handler.get_weight_version({})
        assert status["protocol_version"] == 1
        assert "weight_checksums" not in status
    handler.engine_client.collective_rpc.assert_awaited_once_with(
        "get_weight_transfer_device_info"
    )


@pytest.mark.asyncio
async def test_successful_control_acknowledgements_have_versioned_status(handler):
    await begin(handler)
    status = await handler.start_weight_update({"session_id": "round-1", "seq": 0})
    assert status["protocol_version"] == 1
    assert status["device_uuids"] == ["GPU-test"]
    assert status["model_max_len"] == 128


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        [],
        [{"device_uuid": ""}],
        [{"device_uuid": 1}],
        [{"device_uuid": "GPU-a"}, {"device_uuid": "GPU-b"}],
    ],
)
async def test_invalid_device_identity_never_enters_mutation(handler, reply):
    handler.engine_client.collective_rpc.return_value = reply
    assert (await handler.pause_generation({"mode": "abort"}))["status"] == "error"
    handler.engine_client.pause_generation.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [[], [{}], [{1: "a"}], [{"w": 1}]])
async def test_checksum_diagnostic_rejects_malformed_values(handler, reply):
    await handler.get_weight_version({})
    handler.engine_client.compute_weight_checksums.return_value = reply
    assert (await handler.get_weight_checksums({}))["status"] == "error"


@pytest.mark.asyncio
async def test_engine_death_fences_session_before_shutdown_callback(handler):
    dead = EngineDeadError()
    handler.engine_client.pause_generation.side_effect = dead
    with pytest.raises(RuntimeError, match="dead"):
        await handler.pause_generation({"mode": "abort"})
    assert handler.session.snapshot().failed is True
    handler._shutdown_on_engine_dead.assert_called_once_with(dead)
    with pytest.raises(ValueError, match="failed"):
        await handler.admit_generation()


@pytest.mark.asyncio
async def test_engine_death_still_propagates_if_shutdown_callback_returns(handler):
    dead = EngineDeadError()
    handler._shutdown_on_engine_dead.side_effect = None
    handler.engine_client.pause_generation.side_effect = dead
    with pytest.raises(EngineDeadError):
        await handler.pause_generation({"mode": "abort"})
    assert handler.session.snapshot().failed is True


@pytest.mark.parametrize("limit", [None, True, 0, -1, "128"])
def test_controller_requires_actual_positive_context_limit(handler, limit):
    with pytest.raises(ValueError, match="model_max_len"):
        StrictRLWeightTransferController(
            handler.engine_client, limit, asyncio.Lock(), MagicMock()
        )


@pytest.mark.asyncio
async def test_canonical_routes_receive_versioned_status_without_losing_fields(handler):
    callback = AsyncMock(return_value={"status": "ok", "alive": True})
    route = handler.versioned_route(callback)
    response = await route({})
    assert response["alive"] is True and response["protocol_version"] == 1
    callback.return_value = {"status": "error", "message": "offline"}
    assert await route({}) == callback.return_value
