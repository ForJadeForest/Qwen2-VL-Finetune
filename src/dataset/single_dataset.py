import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src.dataset.sft_dataset import SupervisedDataset

from src.constants import IGNORE_INDEX


from src.params import DataArguments


class SingleDataset(SupervisedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        assert data_args.image_folder is not None, "image_folder must be provided"

        super(SingleDataset, self).__init__(
            data_path=data_path,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            padding=padding,
        )

    def __len__(self):
        return len(self.list_data_dict)

    def next_rand(self):
        import random

        return random.randint(0, len(self) - 1)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = super(SingleDataset, self).__getitem__(i)
        data_dict["level_probs"] = sources.get("level_probs", [-10000] * 5)

        return data_dict


class DataCollatorForSingleDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids,
            padding_side="right",
            padding_value=self.pad_token_id,
            batch_first=True,
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(
            batch_label_ids,
            padding_side="right",
            padding_value=IGNORE_INDEX,
            batch_first=True,
        )

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts
        data_dict["level_probs"] = torch.tensor(
            [example["level_probs"] for example in examples]
        )

        return data_dict


def make_single_data_module(
    model_id, processor: transformers.ProcessorMixin, data_args: DataArguments
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SingleDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
    )
    data_collator = DataCollatorForSingleDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )
    eval_dataset = None
    if data_args.val_path is not None:
        eval_dataset = SingleDataset(
            data_path=data_args.val_path,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
        )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
