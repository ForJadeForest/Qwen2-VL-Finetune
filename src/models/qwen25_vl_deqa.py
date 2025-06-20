from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)
#from transformers.utils import auto_docstring


class Qwen2_5_VLForDEQA(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, weight_softkl=0.1, level_ids=None, level_prefix=None):
        super().__init__(config)
        self.weight_softkl = weight_softkl
        self.level_ids = level_ids
        self.level_prefix = level_prefix

    #@auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        use_softkl_loss: Optional[bool] = True,
        level_probs: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        use_softkl_loss (`bool`, *optional*):
            Whether to use softkl loss.
        level_probs (`torch.Tensor` of shape `(batch_size, 5)`, *optional*):
            The probability of each level.
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss_kl = None    
        if use_softkl_loss and labels is not None:
            loss_kl, idx_level_label, idx_level_logit = self.softkl_loss(
                logits, labels, level_probs
            )
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

            labels_del = del_elements(labels, idx_level_label)
            logits_del = del_elements(logits, idx_level_logit)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if loss_kl is None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            else:
                shift_logits = logits_del[..., :-1, :].contiguous()
                shift_labels = labels_del[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        print(f"loss: {loss}, loss_kl: {loss_kl}, self.weight_softkl: {self.weight_softkl}")
        loss = loss + self.weight_softkl * loss_kl

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def softkl_loss(self, logits, labels, level_probs):
        batch_size = logits.shape[0]
        level_prefix = torch.tensor(self.level_prefix).to(labels.device)
        idx_prefix_label = find_prefix(labels, level_prefix)  # B
        idx_level_label = idx_prefix_label + level_prefix.shape[0]

        level_ids_label = labels[torch.arange(batch_size), idx_level_label]
        for level_id in level_ids_label:
            assert level_id in self.level_ids

        # After padding in prepare_inputs_labels_for_multimodal(), the length of labels will be the same as logits
        assert logits.shape[1] == labels.shape[1]
        idx_level_logit = idx_level_label - 1
        logits_level_ids = logits[
            torch.arange(batch_size), idx_level_logit
        ].contiguous()  # [B, V]

        preds = torch.softmax(logits_level_ids, dim=1)  # [B, V]
        target = torch.zeros_like(preds)  # [B, V]
        target[:, self.level_ids] = level_probs
        target = target.detach()

        pred_log = torch.log(preds)
        loss_kl = F.kl_div(pred_log, target, reduction="batchmean")
        return loss_kl, idx_level_label, idx_level_logit


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
    assert (indices >= 0).all(), "Some inputs do not contain prefix"
    return indices
