import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy,
)
from torch.nn import CrossEntropyLoss
from train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)


def find_prefix(input_ids, prefix):
    """
    input_ids: [B, N1], no start token
    prefix: [N2, ], no start token
    """
    len_prefix = prefix.shape[0]  # N2
    # Create all possible windows of len_prefix
    input_ids_unfold = input_ids.unfold(1, len_prefix, 1)
    # Check if all elements in the window match the sequence
    matches = (input_ids_unfold == prefix).all(dim=2)
    # Convert boolean matches to integers for argmax operation
    matches_int = matches.type(torch.int64)
    # Calculate indices for the first match, if any, otherwise set to -1
    indices = torch.where(
        matches.any(dim=1),
        matches_int.argmax(dim=1),
        torch.tensor(-1, dtype=torch.int64),
    )
    assert (indices >= 0).all(), f"Some inputs do not contain prefix, {input_ids}"
    return indices


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def del_elements(source, idx):
    """source: [B, N] / [B, N, V],
    idx: [B, ] with the value range [0, N-1]"""
    mask = torch.ones([*source.shape[:2]], dtype=torch.bool)
    for idx_1, idx_del in enumerate(idx):
        mask[idx_1, idx_del] = False
    if len(source.shape) == 2:
        source_del = source[mask].view(source.size(0), source.size(1) - 1)
    else:
        assert len(source.shape) == 3
        source_del = source[mask].view(
            source.size(0), source.size(1) - 1, source.size(2)
        )
    return source_del


class QwenDeQATrainer(Trainer):

    def __init__(self, level_prefix, level_ids, use_softkl_loss, weight_softkl, *args, **kwargs):
        super(QwenDeQATrainer, self).__init__(*args, **kwargs)
        self.level_prefix = level_prefix
        self.level_ids = level_ids
        self.use_softkl_loss = use_softkl_loss
        self.loss_fct = CrossEntropyLoss()
        self.weight_softkl = weight_softkl

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [
                    name for name, _ in opt_model.named_parameters() if "merger" in name
                ]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(), require_grad_only=False
            )
            torch.save(
                non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin")
            )

            if (
                self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]
                and self.state.best_global_step
            ):
                best_checkpoint_folder = (
                    f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                )
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb
                    for cb in self.callback_handler.callbacks + [self.control]
                    if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super(QwenDeQATrainer, self)._save_checkpoint(model, trial)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        level_probs = inputs.pop("level_probs", None)
        assert level_probs is not None
        labels = inputs.pop("labels", None)
        assert labels is not None
        output = model(**inputs, return_dict=True)
        logits = output.logits
        assert output.loss is None, f"loss is not None, {output.loss}"
        if self.use_softkl_loss:
            loss_kl, idx_level_label, idx_level_logit = self.softkl_loss(
                logits, labels, level_probs
            )
            labels_del = del_elements(labels, idx_level_label)
            logits_del = del_elements(logits, idx_level_logit)
            shift_logits = logits_del[..., :-1, :].contiguous()
            shift_labels = labels_del[..., 1:].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        vocab_size = self.model.config.get_text_config().vocab_size
        assert vocab_size is not None, f"vocab_size is not found in model.config, {self.model.config}"
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        ce_loss = self.loss_fct(shift_logits, shift_labels)

        loss = ce_loss

        if self.use_softkl_loss:
            loss = loss + self.weight_softkl * loss_kl

        if return_outputs:
            return (loss,) + tuple(output)
        return loss

    def softkl_loss(self, logits, labels, level_probs):
        batch_size = logits.shape[0]
        level_prefix = torch.tensor(self.level_prefix).to(labels.device)
        idx_prefix_label = find_prefix(labels, level_prefix)  # B
        idx_level_label = idx_prefix_label + level_prefix.shape[0]

        level_ids_label = labels[torch.arange(batch_size), idx_level_label]

        for level_id in level_ids_label:
            assert level_id in self.level_ids, f"level_id: {level_id} not in level_ids: {self.level_ids}"
        # After padding in prepare_inputs_labels_for_multimodal(), the length of labels will be the same as logits
        assert logits.shape[1] == labels.shape[1]
        idx_level_logit = idx_level_label - 1
        logits_level_ids = logits[
            torch.arange(batch_size), idx_level_logit
        ].contiguous()  # [B, V]

        # Use log_softmax for numerical stability
        log_preds = F.log_softmax(logits_level_ids, dim=1)  # [B, V]

        # Gather the log probabilities of the target levels
        log_preds_at_levels = log_preds[:, self.level_ids]  # [B, K]

        # Manually compute KL divergence to avoid log(0) on the full target vector.
        # KL(p || q) = sum(p * (log p - log q))
        # We assume level_probs > 0. A small epsilon is added for stability.
        log_level_probs = torch.log(level_probs.clamp(min=1e-9))

        # The F.kl_div function expects input=log_q, target=p.
        # The loss is sum(p * (log p - log q)).
        # PyTorch's kl_div with reduction='batchmean' divides by batch_size.
        # We will compute sum over distribution, and then mean over batch.
        kl_div_per_element = level_probs * (log_level_probs - log_preds_at_levels)
        loss_kl = kl_div_per_element.sum(dim=-1).mean()

        assert not torch.isnan(loss_kl)

        return loss_kl, idx_level_label, idx_level_logit
