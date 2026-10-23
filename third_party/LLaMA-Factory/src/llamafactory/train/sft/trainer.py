# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import math
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        self._eval_sft_entropy_sum = 0.0
        self._eval_sft_entropy_count = 0.0

    def _record_sft_entropy(self, logits: "torch.Tensor", labels: "torch.Tensor", active_mask: Optional["torch.Tensor"] = None) -> None:
        if not self.finetuning_args.log_sft_entropy:
            return

        if self.model.training:
            return

        if labels is None or logits is None or logits.dim() != 3:
            return

        with torch.no_grad():
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            if active_mask is None:
                active_mask = shift_labels != IGNORE_INDEX
            else:
                active_mask = active_mask.to(logits.device)

            active_indices = active_mask.view(-1).nonzero(as_tuple=False).flatten()
            if active_indices.numel() == 0:
                return

            sample_size = int(getattr(self.finetuning_args, "log_sft_entropy_sample_size", 512))
            if sample_size > 0 and active_indices.numel() > sample_size:
                pick = torch.linspace(
                    0,
                    active_indices.numel() - 1,
                    steps=sample_size,
                    device=active_indices.device,
                ).round().long()
                active_indices = active_indices[pick]

            vocab_size = logits.size(-1)
            seq_len = shift_labels.size(1)
            entropy_sum = logits.new_zeros((), dtype=torch.float32)
            chunk_size = 128

            for start in range(0, active_indices.numel(), chunk_size):
                chunk_indices = active_indices[start : start + chunk_size]
                batch_indices = torch.div(chunk_indices, seq_len, rounding_mode="floor")
                token_indices = chunk_indices.remainder(seq_len)
                chunk_logits = logits[batch_indices, token_indices, :].float()
                chunk_log_probs = F.log_softmax(chunk_logits, dim=-1)
                chunk_probs = chunk_log_probs.exp()
                chunk_entropy = -(chunk_probs * chunk_log_probs).sum(dim=-1)
                entropy_sum = entropy_sum + chunk_entropy.sum()

            self._eval_sft_entropy_sum += float(entropy_sum.detach().cpu())
            self._eval_sft_entropy_count += float(active_indices.numel())

    def _pop_eval_sft_entropy_logs(self) -> dict[str, float]:
        entropy_sum, entropy_count = self._eval_sft_entropy_sum, self._eval_sft_entropy_count
        self._eval_sft_entropy_sum = 0.0
        self._eval_sft_entropy_count = 0.0

        if entropy_count <= 0:
            return {}

        stats = torch.tensor([entropy_sum, entropy_count], dtype=torch.float64, device=self.args.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        entropy = (stats[0] / stats[1]).item()
        vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", None)
        logs = {"eval_sft_entropy": entropy}
        if vocab_size and vocab_size > 1:
            logs["eval_sft_entropy_norm"] = entropy / math.log(vocab_size)

        return logs

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self.finetuning_args.log_sft_entropy and "eval_loss" in logs:
            logs.update(self._pop_eval_sft_entropy_logs())

        return super().log(logs, *args, **kwargs)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        loss_weights = inputs.pop("loss_weights", None)
        if loss_weights is None or not torch.any(loss_weights > 0):
            if self.finetuning_args.log_sft_entropy and "labels" in inputs:
                outputs = model(**inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                self._record_sft_entropy(logits, inputs.get("labels"))
                return (loss, outputs) if return_outputs else loss

            return super().compute_loss(model, inputs, return_outputs, *args, **kwargs)

        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs, *args, **kwargs)

        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        shift_labels = labels[..., 1:].contiguous().to(logits.device)
        shift_weights = loss_weights[..., 1:].contiguous().to(logits.device).float()
        active_mask = (shift_labels != IGNORE_INDEX) & (shift_weights > 0)
        denom = shift_weights[active_mask].sum()
        if denom <= 0:
            loss = logits.sum() * 0.0
        else:
            vocab_size = logits.size(-1)
            seq_len = shift_labels.size(1)
            active_indices = active_mask.view(-1).nonzero(as_tuple=False).flatten()
            flat_labels = shift_labels.view(-1)
            flat_weights = shift_weights.view(-1)
            loss_sum = logits.new_zeros((), dtype=torch.float32)
            for start in range(0, active_indices.numel(), 2048):
                chunk_indices = active_indices[start : start + 2048]
                batch_indices = torch.div(chunk_indices, seq_len, rounding_mode="floor")
                token_indices = chunk_indices.remainder(seq_len)
                chunk_logits = logits[batch_indices, token_indices, :].view(-1, vocab_size)
                chunk_labels = flat_labels[chunk_indices]
                chunk_weights = flat_weights[chunk_indices]
                chunk_loss = F.cross_entropy(chunk_logits, chunk_labels, reduction="none")
                loss_sum = loss_sum + (chunk_loss.float() * chunk_weights).sum()

            loss = loss_sum / denom

        self._record_sft_entropy(logits, labels, active_mask=active_mask)
        return (loss, outputs) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
