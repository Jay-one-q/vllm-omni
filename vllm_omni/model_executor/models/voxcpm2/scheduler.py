# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler
from vllm_omni.core.sched.output import OmniNewRequestData

logger = init_logger(__name__)


class VoxCPM2OmniARAsyncScheduler(OmniARAsyncScheduler):
    """VoxCPM2 scheduler variant for full unified decode graph serving.

    VoxCPM2's full unified decode graph only applies to pure decode batches.
    When a decode-ready request is already running, this scheduler defers new
    waiting admissions for the current tick so the decode batch can stay on the
    VoxCPM2 graph path. This is a model-local serving policy, not a generic AR
    scheduler rule.
    """

    def _unified_decode_graph_enabled(self) -> bool:
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        runtime_config = getattr(hf_config, "voxcpm2_runtime_config", None)
        if isinstance(runtime_config, dict):
            return bool(runtime_config.get("enable_unified_decode_graph", False))
        return bool(getattr(runtime_config, "enable_unified_decode_graph", False))

    def _should_defer_waiting_for_unified_decode_graph(self) -> bool:
        if not self._unified_decode_graph_enabled():
            return False
        if not self.waiting or not self.running:
            return False

        for request in self.running:
            if getattr(request, "status", None) != RequestStatus.RUNNING or request.is_finished():
                continue
            if self._get_confirmed_num_computed_tokens(request) >= request.num_prompt_tokens:
                return True
        return False

    def schedule(self) -> SchedulerOutput:  # type: ignore[override]
        defer_waiting = self._should_defer_waiting_for_unified_decode_graph()

        # Keep the common AR scheduler path untouched, but preserve its exact
        # ordering here. In particular, the deferred waiting queue must be
        # restored immediately after the upstream schedule call, before the
        # omni chunk/input coordinators restore their own queues.
        for queue in (self.waiting, self.running):
            for req in list(queue):
                if getattr(req, "status", None) == RequestStatus.FINISHED_ABORTED:
                    queue.remove(req)
        self._consume_pending_connector_output(model_mode="ar")
        self._process_pending_input_timeouts()

        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(
                self.waiting, self.running, scheduler_requests=self.requests
            )

        original_waiting = None
        if defer_waiting:
            original_waiting = self.waiting
            self.waiting = create_request_queue(self.policy)

        try:
            scheduler_output = super(OmniARScheduler, self).schedule()
        finally:
            if original_waiting is not None:
                deferred_waiting = list(self.waiting)
                if deferred_waiting:
                    original_waiting.prepend_requests(deferred_waiting)
                self.waiting = original_waiting
            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.restore_queues(
                    self.waiting,
                    self.running,
                    scheduler_requests=self.requests,
                )
            if self.input_coordinator:
                self.input_coordinator.restore_queues(self.waiting, self.running)

        try:
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                new_list.append(
                    OmniNewRequestData(
                        req_id=nr.req_id,
                        external_req_id=(getattr(request, "external_req_id", None) if request else None),
                        prompt_token_ids=nr.prompt_token_ids,
                        mm_features=nr.mm_features,
                        sampling_params=nr.sampling_params,
                        pooling_params=nr.pooling_params,
                        block_ids=nr.block_ids,
                        num_computed_tokens=nr.num_computed_tokens,
                        lora_request=nr.lora_request,
                        prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                        prompt_is_token_ids=nr.prompt_is_token_ids,
                        additional_information=(getattr(request, "additional_information", None) if request else None),
                    )
                )

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output, self.requests)
            finished_reqs = self.get_finished_requests_needing_kv_transfer()
        except Exception:
            logger.exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")
            finished_reqs = {}

        return self._wrap_omni_scheduler_output(
            scheduler_output,
            finished_requests_needing_kv_transfer=finished_reqs,
        )
