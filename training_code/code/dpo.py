# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, "transformers-4.41-release", 'src')))

import json

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import logging

import torch
import transformers
import argparse
import shutil
from transformers import set_seed

from transformers import AutoTokenizer


from lora_utils import (
    SavePeftModelCallback,
    print_trainable_parameters,
    get_last_checkpoint,
    DEFAULT_PAD_TOKEN,
)
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.train.train import smart_tokenizer_and_embedding_resize
from data_utils.common_utils import preprocess
from data_utils.data_utils_dpo import *

from models.trainer.dpo_trainer import *

from models.config.dpo_config import DPOConfig

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    base_model_name_or_path: Optional[str] = field(default=None)
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    # from LLaVA
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default=None, metadata={"help": "Dataset name"})
    eval_dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    eval_dataset_name: str = field(default="alpaca_human_preference")
    eval_size: int = field(
        default=500,
        metadata={
            "help": "Number of examples to split out from training to use for evaluation."
        },
    )
    # From LLaVA
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    reward_prompt_file: Optional[str] = field(default=None)
    image_to_caption_file: Optional[str] = field(default=None)

    dataset_format: Optional[str] = field(default="v1_UVQA")
    source_max_len: Optional[int] = field(default=512)
    target_max_len: Optional[int] = field(default=512)

    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)

    num_frames: Optional[int] = field(default=8)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # From LLaVA
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    # From AlpacaFarm
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    query_len: int = field(default=None, metadata={"help": "Length of the query."})
    response_len: int = field(
        default=None, metadata={"help": "Length of the response."}
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice", "labels"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    # From QLoRA
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Which modules to use LoRA on. If None, will use all linear layers."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoint to resume."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_training: bool = field(
        default=False, metadata={"help": "Resume from training"}
    )
    torch_dtype: str = field(
        default="bfloat16", metadata={"help": "The torch dtype to use for the model"}
    )
    temperature: float = field(
        default=0.2, metadata={"help": "Temperature for sampling from the model"}
    )
    generate_during_eval: bool = field(
        default=False, metadata={"help": "Generate during evaluation"}
    )
    model_adapter_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the adapter to use"}
    )
    running_model_name: Optional[str] = field(
        default="vlm-rlaif", metadata={"help": "The name of the model to train: vlm-rlaif, llama-vid, video-llava"}
    ) #
    dataset_setting: Optional[str] = field(
        default="uvqa_relation+uvqa_object", metadata={"help": "The name of the dataset setting"}
    )
    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "The maximum number of steps to train"}
    )
    ref_adapter_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the reference adapter to use"}
    )
    disable_dropout: bool = field(
        default=True, metadata={"help": "Disable dropout in the model"}
    )

def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, DPOConfig)
    )
    (
        model_args,
        data_args,
        training_args,
        dpo_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(dpo_args)
    )
    setattr(data_args, "running_model_name", args.running_model_name)



    if args.resume_dir is not None:
        checkpoint_dir, completed_training = args.resume_dir, False
    else:
        checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)

    if completed_training:
        rank0_print("Detected that training was already completed!")

    if checkpoint_dir is None:
        rank0_print("Training from scratch.")
    else:
        rank0_print("Loading from checkpoint:", checkpoint_dir)
        if args.resume_from_training:
            rank0_print("Resuming from training not supported yet. Exiting.")
            exit(1)
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )

    tokenizer_model_name = args.model_name_or_path
    TokenizerClass = AutoTokenizer

    # Tokenizer
    if args.running_model_name == "vlm-rlaif":
        cfg = None
        tokenizer = TokenizerClass.from_pretrained(
            pretrained_model_name_or_path=tokenizer_model_name,
            cache_dir=args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            truncation_side="right",
            use_fast=False,
        )

        if model_args.version == "v0":
            if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="[PAD]"),
                    tokenizer=tokenizer,
                    model=model,
                )
        elif model_args.version == "v0.5":
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
            if model_args.version in conversation_lib.conv_templates:
                conversation_lib.default_conversation = conversation_lib.conv_templates[
                    model_args.version
                ]
            else:
                conversation_lib.default_conversation = conversation_lib.conv_templates[
                    "vicuna_v1"
                ]

        
        if model_args.vision_tower is not None:
            from llava.model import LlavaLlamaForCausalLM 
            with DisableLogger():
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    torch_dtype=torch_dtype,
                    trust_remote_code=model_args.trust_remote_code,

                )

                ref_model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    torch_dtype=torch_dtype,
                    trust_remote_code=model_args.trust_remote_code,
                )

            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            
            vision_tower = ref_model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()

            data_args.image_processor = vision_tower.image_processor
            data_args.is_multimodal = True
            model.config.mm_use_im_start_end = (
                data_args.mm_use_im_start_end
            ) = model_args.mm_use_im_start_end
            training_args.use_im_start_end = model_args.mm_use_im_start_end

    else:
        print("not implemented yet")
        exit(1)        
        

    data_module = make_dpo_data_module(
        tokenizer=tokenizer, 
        args=args, 
        data_args=data_args,
        model=model,
        config=cfg,
        )

    if args.do_train:
        training_data = data_module['train_dataset']
        rank0_print("Training data size:", len(training_data))
        rank0_print("Training data example:")
        if args.running_model_name == "video_chat2":
            for i in range(min(3, len(training_data))):
                rank0_print(training_data[i][1])
                rank0_print("=" * 20)
        else:
            for i in range(min(3, len(training_data))):
                ex_input_ids_0 = training_data[i]["chosen_input_ids"]
                ex_input_ids_0[ex_input_ids_0 == IMAGE_TOKEN_INDEX] = tokenizer.eos_token_id
                rank0_print(tokenizer.decode(ex_input_ids_0, skip_special_tokens=False))
                rank0_print("=" * 20)
                ex_input_ids_1 = training_data[i]["chosen_input_ids"]
                ex_input_ids_1[ex_input_ids_1 == IMAGE_TOKEN_INDEX] = tokenizer.eos_token_id
                rank0_print(
                    tokenizer.decode(
                        ex_input_ids_1,
                        skip_special_tokens=False,
                    )
                )
                rank0_print("=" * 20)
                rank0_print("=" * 20)
    if args.running_model_name == "video_chat2":
        pass
    else:
        model.config.use_cache = False
        ref_model.config.use_cache = False

    print_trainable_parameters(args, model)
    print("loaded model")
    set_seed(args.seed)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=training_data,
        eval_dataset=data_module['eval_dataset'],
        data_collator=data_module['train_data_collator'],
        eval_data_collator=data_module['eval_data_collator'],
        args=training_args,
        all_args=args,
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if args.do_train or args.do_eval:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))
    
    # trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()