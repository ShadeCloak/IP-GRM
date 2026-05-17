# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reward model worker for generative and discriminative scoring."""

import logging
import os

import torch

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import (
    get_device_name,
    get_torch_device,
)
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import DistProfiler, DistProfilerExtension, log_gpu_memory_usage
from verl.workers.config import HFModelConfig, RewardModelConfig, RewardModelDataProcessorConfig
from verl.workers.roles.reward_model_engine import get_reward_model_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class RewardModelWorker(Worker, DistProfilerExtension):
    def __init__(self, config: RewardModelConfig) -> None:
        self.config = config
        self.model_config = config.model_config
        self.input_model_config = config.input_model_config
        self.model_type = config.model_type
        assert self.model_type in ["discriminative", "generative"], f"model_type: {self.model_type} is not supported"

        # Check if self-evaluation penalty is enabled (reward hacking prevention)
        self.enable_self_eval_penalty = config.get("enable_self_eval_penalty", False)
        if self.enable_self_eval_penalty:
            logger.info("Self-evaluation penalty enabled (code-level reward hacking prevention)")

        # Check if pairwise_v1 mode is enabled
        self.pairwise_v1 = config.get("pairwise_v1", False)
        if self.pairwise_v1:
            logger.info("Pairwise V1 mode enabled for reward model")

        Worker.__init__(self)
        self.profiler_config = self.config.profiler
        tool_config = self.profiler_config.tool_config
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config)
        )

        initialize_global_process_group_ray(timeout_second=None)

    def _build_reward_model(self):
        from torch.distributed.device_mesh import init_device_mesh
        import importlib.util
        from transformers import AutoTokenizer

        reward_model_config: RewardModelConfig = self.config
        model_config: HFModelConfig = self.config.model_config
        self.data_processor_config = self.config.data_processor_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.path, trust_remote_code=True)

        # Resolve source tokenizer (actor model tokenizer when different from RM)
        input_model_path = None
        if self.input_model_config is not None:
            input_model_path = self.input_model_config.get("path", None)

        if input_model_path is None or input_model_path == "":
            self._do_switch_chat_template = False
            self.src_tokenizer = self.tokenizer
        else:
            self._do_switch_chat_template = True
            if hasattr(self.input_model_config, 'get_processor'):
                self.src_tokenizer = self.input_model_config.get_processor()
            else:
                self.src_tokenizer = AutoTokenizer.from_pretrained(input_model_path, trust_remote_code=True)

        # Dynamically load process functions from config path
        def load_fn(py_file, fn_name):
            spec = importlib.util.spec_from_file_location("data_proc", py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, fn_name)

        dp_cfg = self.data_processor_config
        self.preprocess_fn = load_fn(dp_cfg.path, dp_cfg.preprocess_fn_name)
        self.postprocess_fn = load_fn(dp_cfg.path, dp_cfg.postprocess_fn_name)

        if self.pairwise_v1:
            self.pairwise_preprocess_fn = load_fn(
                dp_cfg.path, dp_cfg.get("pairwise_preprocess_fn_name", "construct_deepseek_grm_inputs_pairwise"))
            self.pairwise_postprocess_fn = load_fn(
                dp_cfg.path, dp_cfg.get("pairwise_postprocess_fn_name", "convert_deepseek_grm_pairwise_output_to_comparison"))

        self.two_stage_grm = self.config.get("two_stage_grm", True)
        if self.two_stage_grm:
            self.construct_principles_fn = load_fn(dp_cfg.path, "construct_principles_only_input")
            self.construct_judge_fn = load_fn(dp_cfg.path, "construct_judge_with_prefix_input")
            self.extract_principles_fn = load_fn(dp_cfg.path, "extract_principles_from_output")
            logger.info("Two-stage GRM enabled")

        if self.model_type == "generative":
            assert self.preprocess_fn is not None and self.postprocess_fn is not None, (
                "generative reward model must have preprocess_fn and postprocess_fn"
            )

        # Build device mesh
        infer_tp = self.config.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"world_size {self.world_size} not divisible by infer_tp {infer_tp}"
        )
        reward_model_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        self._rm_device_mesh = reward_model_device_mesh

        is_collect = reward_model_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "reward_model", dp_rank=reward_model_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # Init random states (ensure all TP ranks share the same state)
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = reward_model_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # Build reward model
        log_gpu_memory_usage("Before building reward model", logger=logger)
        self.reward_model = get_reward_model_class(reward_model_config.name)(
            config=reward_model_config, model_config=model_config, device_mesh=reward_model_device_mesh
        )
        log_gpu_memory_usage("After building reward model", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._build_reward_model()

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _preprocess_reward_inputs(self, data: DataProto):
        src_tokenizer = self.src_tokenizer
        tokenizer = self.tokenizer
        rm_inputs = []
        self._original_questions = []
        self._original_responses = []

        for i in range(len(data)):
            data_item = data[i]

            # Extract question
            if "extra_infos" in data_item.non_tensor_batch and "question" in data_item.non_tensor_batch["extra_infos"]:
                rollout_question = data_item.non_tensor_batch["extra_infos"]["question"]
            else:
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                rollout_question = src_tokenizer.decode(valid_prompt_ids.tolist(), skip_special_tokens=True)

            # Extract response
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            rollout_response = src_tokenizer.decode(valid_response_ids.tolist(), skip_special_tokens=True)

            self._original_questions.append(rollout_question)
            self._original_responses.append(rollout_response)

            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)

            if self.model_type == "discriminative":
                if self._do_switch_chat_template:
                    chats = [
                        {"role": "user", "content": rollout_question},
                        {"role": "assistant", "content": rollout_response},
                    ]
                    rm_input = tokenizer.apply_chat_template(chats, tokenize=True)
                else:
                    non_pad_indices = torch.nonzero(data_item.batch["attention_mask"], as_tuple=True)[0]
                    start_idx, end_idx = non_pad_indices[0], non_pad_indices[-1]
                    rm_input = data_item.batch["input_ids"][start_idx : end_idx + 1].tolist()
            else:
                assert self.preprocess_fn is not None

                preprocess_kwargs = {
                    "rollout_question": rollout_question,
                    "rollout_response": rollout_response,
                    "ground_truth": ground_truth,
                }
                # Pass cached principles if available (two_stage_grm=False fallback)
                principals = data_item.non_tensor_batch.get("principals", None)
                if principals is not None:
                    preprocess_kwargs["principals"] = principals

                input_str = self.preprocess_fn(**preprocess_kwargs)

                # Pass string directly for generative RM (vLLM handles tokenization)
                if "DeepSeek-GRM" in self.model_config.path:
                    rm_input = input_str
                else:
                    chats = [{"role": "user", "content": input_str}]
                    rm_input = tokenizer.apply_chat_template(chats, add_generation_prompt=True, tokenize=True)

            rm_inputs.append(rm_input)

        return rm_inputs

    def _postprocess_reward_outputs(self, data: DataProto, output: list[float] | list[list[int]] | list[str]):
        import inspect
        import numpy as np

        output_lengths_chars = []
        output_lengths_tokens = []
        extraction_methods = []
        failure_cases = []
        METHODS_TO_LOG = ["failed_return_0", "exception_return_0", "fallback_score_colon", "fallback_last_number"]

        if self.model_type == "discriminative":
            scores = torch.tensor(output)
        else:
            assert self.postprocess_fn is not None

            if output and isinstance(output[0], str):
                output_text = output
            else:
                output_text = [self.tokenizer.decode(o) for o in output]

            if hasattr(self.reward_model, '_last_token_counts') and self.reward_model._last_token_counts:
                output_lengths_tokens = self.reward_model._last_token_counts
                self.reward_model._last_token_counts = None
            else:
                output_lengths_tokens = []

            supports_metadata = 'return_metadata' in inspect.signature(self.postprocess_fn).parameters

            if supports_metadata:
                results = [self.postprocess_fn(o, return_metadata=True) for o in output_text]
                scores = [r["score"] if isinstance(r, dict) else r for r in results]
                for idx, r in enumerate(results):
                    if isinstance(r, dict):
                        output_lengths_chars.append(r.get("output_length", len(output_text[idx])))
                        method = r.get("extraction_method", "unknown")
                        extraction_methods.append(method)
                        if method in METHODS_TO_LOG:
                            failure_cases.append({
                                "index": idx, "extraction_method": method, "score": r["score"],
                                "question": self._original_questions[idx] if hasattr(self, '_original_questions') and idx < len(self._original_questions) else "N/A",
                                "response": self._original_responses[idx] if hasattr(self, '_original_responses') and idx < len(self._original_responses) else "N/A",
                                "rm_output": output_text[idx],
                                "output_length": r.get("output_length", len(output_text[idx])),
                            })
                    else:
                        output_lengths_chars.append(len(output_text[idx]) if output_text else 0)
                        extraction_methods.append("no_metadata")
            else:
                scores = [self.postprocess_fn(o) for o in output_text]
                output_lengths_chars = [len(o) for o in output_text]
                extraction_methods = ["legacy_no_metadata"] * len(output_text)

            # Handle None scores (invalid RM outputs)
            valid_mask = [s is not None for s in scores]
            num_invalid = sum(1 for v in valid_mask if not v)
            if num_invalid > 0:
                logger.warning(f"{num_invalid}/{len(scores)} responses with invalid reward scores")
                scores = [s if s is not None else 0.0 for s in scores]
                self._invalid_mask = np.array([not v for v in valid_mask], dtype=bool)
            else:
                self._invalid_mask = None

            # Self-evaluation penalty (reward hacking prevention)
            if self.enable_self_eval_penalty:
                self._apply_self_eval_penalty(scores)

            scores = torch.tensor(scores)

        token_level_scores = self._expand_to_token_level(data, scores)
        self._last_output_metadata = {
            "output_lengths_chars": output_lengths_chars,
            "output_lengths_tokens": output_lengths_tokens,
            "extraction_methods": extraction_methods,
            "failure_cases": failure_cases,
        }
        return token_level_scores

    def _apply_self_eval_penalty(self, scores: list):
        """Penalize responses containing self-grading patterns (reward hacking prevention)."""
        import re
        SELF_EVAL_PATTERNS = [
            "特点总结：", "评分（满分10分）：", "综合评分：", "总评：", "总分：",
            "评分：", "得分：", "打分：", "分数：",
            "Score (on 10-point scale):", "Tone & Style Summary:", "Overall Rating:",
            "Overall Score:", "Final Score:", "Total Score:",
            "/10 —", "/10—", "/10（", "/10 (",
        ]
        SCORE_RE = re.compile(r'\b[89]\.\d/10\b|\b10/10\b|\b9\.[5-9]/10\b')
        PENALTY_SCORE = 3.0

        if not hasattr(self, '_original_responses') or not self._original_responses:
            return

        num_penalized = 0
        for idx, response in enumerate(self._original_responses):
            if idx >= len(scores):
                break
            tail = response[-2000:]
            matched = next((p for p in SELF_EVAL_PATTERNS if p in tail), None)
            if not matched and SCORE_RE.search(tail):
                matched = "score_pattern_regex"
            if matched:
                original = scores[idx]
                scores[idx] = PENALTY_SCORE
                num_penalized += 1
                if num_penalized <= 3:
                    logger.warning(f"[SELF-EVAL PENALTY] #{idx}: '{matched}', {original:.1f} -> {PENALTY_SCORE}")

        if num_penalized > 0:
            logger.warning(f"[SELF-EVAL PENALTY] {num_penalized}/{len(scores)} responses penalized")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward_model"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        import numpy as np
        from collections import Counter

        data = data.to("cpu")

        if self.pairwise_v1:
            token_level_scores = self._compute_pairwise_scores_with_gather(data)
        elif self.two_stage_grm:
            token_level_scores = self._compute_two_stage_grm_scores(data)
        else:
            rm_data = self._preprocess_reward_inputs(data)
            output = self.reward_model.compute_reward(rm_data)
            token_level_scores = self._postprocess_reward_outputs(data, output)

        metadata = {}
        meta_info = {}

        # Invalid mask for downstream filtering
        if hasattr(self, '_invalid_mask') and self._invalid_mask is not None:
            metadata["invalid_mask"] = self._invalid_mask
            meta_info["num_invalid_scores"] = int(np.sum(self._invalid_mask))
            self._invalid_mask = None
        else:
            batch_size = data.batch.batch_size[0]
            metadata["invalid_mask"] = np.zeros(batch_size, dtype=bool)
            meta_info["num_invalid_scores"] = 0

        if hasattr(self, '_last_output_metadata') and self._last_output_metadata:
            om = self._last_output_metadata
            output_lengths_chars = om.get("output_lengths_chars", [])
            output_lengths_tokens = om.get("output_lengths_tokens", [])
            extraction_methods = om.get("extraction_methods", [])
            failure_cases = om.get("failure_cases", [])

            if output_lengths_chars:
                metadata["rm_output_lengths_chars"] = output_lengths_chars

            output_lengths = output_lengths_tokens or output_lengths_chars
            if output_lengths:
                metadata["rm_output_lengths"] = output_lengths
                metadata["rm_extraction_methods"] = extraction_methods
                meta_info["rm_output_length_min"] = int(np.min(output_lengths))
                meta_info["rm_output_length_mean"] = float(np.mean(output_lengths))
                meta_info["rm_output_length_max"] = int(np.max(output_lengths))
                meta_info["rm_extraction_method_counts"] = dict(Counter(extraction_methods))

            if failure_cases:
                meta_info["rm_failure_cases"] = failure_cases

            self._last_output_metadata = None

        return DataProto.from_dict(
            tensors={"rm_scores": token_level_scores},
            non_tensors=metadata if metadata else {},
            meta_info=meta_info if meta_info else {},
        )

    def _compute_two_stage_grm_scores(self, data: DataProto):
        """Two-stage scoring: generate principles per question, then score each response."""
        from collections import defaultdict

        batch_size = data.batch.batch_size[0]
        src_tokenizer = self.src_tokenizer
        strip_think_tag = self.data_processor_config.get("strip_think_tag", True)

        # Step 1: Extract questions/responses and group by question
        questions, responses = [], []
        question_to_indices = defaultdict(list)
        self._original_questions = []
        self._original_responses = []

        for i in range(batch_size):
            data_item = data[i]
            if "extra_infos" in data_item.non_tensor_batch and "question" in data_item.non_tensor_batch["extra_infos"]:
                question = data_item.non_tensor_batch["extra_infos"]["question"]
            else:
                prompt_ids = data_item.batch["prompts"]
                valid_len = int(data_item.batch["attention_mask"][:prompt_ids.shape[-1]].sum().item())
                question = src_tokenizer.decode(prompt_ids[-valid_len:].tolist(), skip_special_tokens=True)

            resp_ids = data_item.batch["responses"]
            valid_len = int(data_item.batch["attention_mask"][-resp_ids.shape[-1]:].sum().item())
            response = src_tokenizer.decode(resp_ids[:valid_len].tolist(), skip_special_tokens=True)

            questions.append(question)
            responses.append(response)
            question_to_indices[question].append(i)
            self._original_questions.append(question)
            self._original_responses.append(response)

        # Step 2: Stage 1 — generate principles (one call per unique question)
        unique_questions = list(question_to_indices.keys())
        stage1_inputs = [self.construct_principles_fn(rollout_question=q) for q in unique_questions]
        logger.info(f"Stage 1: generating principles for {len(unique_questions)} unique questions")
        stage1_outputs = self.reward_model.compute_reward(stage1_inputs)

        stage1_texts = stage1_outputs if (stage1_outputs and isinstance(stage1_outputs[0], str)) else \
            [self.tokenizer.decode(o) for o in stage1_outputs]
        question_to_principles = {q: self.extract_principles_fn(t) for q, t in zip(unique_questions, stage1_texts)}

        # Step 3: Stage 2 — score each response with shared principles
        stage2_inputs = [
            self.construct_judge_fn(
                rollout_question=questions[i], rollout_response=responses[i],
                principles_prefix=question_to_principles[questions[i]], strip_think_tag=strip_think_tag,
            )
            for i in range(batch_size)
        ]
        logger.info(f"Stage 2: scoring {batch_size} responses")
        stage2_outputs = self.reward_model.compute_reward(stage2_inputs)

        return self._postprocess_reward_outputs(data, stage2_outputs)

    def _compute_pairwise_scores_with_gather(self, data: DataProto):
        """Pairwise scoring with optional all_gather across DP workers."""
        if not hasattr(self, '_rm_device_mesh') or self._rm_device_mesh is None:
            return self._compute_pairwise_scores(data)

        dp_mesh = self._rm_device_mesh["dp"]
        dp_size, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()

        if dp_size <= 1:
            return self._compute_pairwise_scores(data)
        elif dp_size <= 2:
            from verl.protocol import all_gather_data_proto
            local_batch_size = data.batch.batch_size[0]
            all_gather_data_proto(data, process_group=dp_mesh.get_group())
            full_scores = self._compute_pairwise_scores(data)
            return full_scores[dp_rank * local_batch_size : (dp_rank + 1) * local_batch_size]
        else:
            logger.warning(f"Pairwise: dp_size={dp_size}, using local-only (groups may be incomplete)")
            return self._compute_pairwise_scores(data)

    def _compute_pairwise_scores(self, data: DataProto):
        """Pairwise comparison: random baseline per group, +1/-1 rewards."""
        import random
        from collections import defaultdict

        batch_size = data.batch.batch_size[0]

        if "uid" in data.non_tensor_batch:
            indices = data.non_tensor_batch["uid"]
        elif "index" in data.non_tensor_batch:
            indices = data.non_tensor_batch["index"]
        else:
            indices = list(range(batch_size))

        groups = defaultdict(list)
        for i in range(batch_size):
            groups[indices[i]].append(i)

        scores = torch.zeros(batch_size)

        for group_idx, item_indices in groups.items():
            if len(item_indices) < 2:
                scores[item_indices[0]] = 0.0
                continue

            baseline_idx = random.choice(item_indices)
            baseline_item = data[baseline_idx]

            if "extra_infos" in baseline_item.non_tensor_batch and "question" in baseline_item.non_tensor_batch["extra_infos"]:
                baseline_question = baseline_item.non_tensor_batch["extra_infos"]["question"]
            else:
                prompt_ids = baseline_item.batch["prompts"]
                valid_len = int(baseline_item.batch["attention_mask"][:prompt_ids.shape[-1]].sum().item())
                baseline_question = self.src_tokenizer.decode(prompt_ids[-valid_len:].tolist(), skip_special_tokens=True)

            resp_ids = baseline_item.batch["responses"]
            valid_len = int(baseline_item.batch["attention_mask"][-resp_ids.shape[-1]:].sum().item())
            baseline_response = self.src_tokenizer.decode(resp_ids[:valid_len].tolist(), skip_special_tokens=True)

            scores[baseline_idx] = -1.0

            pairwise_inputs, compare_indices = [], []
            for idx in item_indices:
                if idx == baseline_idx:
                    continue
                cur = data[idx]
                r_ids = cur.batch["responses"]
                v_len = int(cur.batch["attention_mask"][-r_ids.shape[-1]:].sum().item())
                cur_response = self.src_tokenizer.decode(r_ids[:v_len].tolist(), skip_special_tokens=True)
                pairwise_inputs.append(self.pairwise_preprocess_fn(
                    rollout_question=baseline_question, response1=cur_response, response2=baseline_response))
                compare_indices.append(idx)

            if pairwise_inputs:
                outputs = self.reward_model.compute_reward(pairwise_inputs)
                output_text = outputs if (outputs and isinstance(outputs[0], str)) else \
                    [self.tokenizer.decode(o) for o in outputs]
                for out, idx in zip(output_text, compare_indices):
                    scores[idx] = float(self.pairwise_postprocess_fn(out))

        # Expand to token level
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores
        return token_level_scores[:, -response_length:]
