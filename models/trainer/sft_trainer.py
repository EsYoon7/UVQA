import dataclasses
import inspect
import warnings
from collections import defaultdict
from functools import wraps
import random
import wandb
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Mapping, Literal

import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from contextlib import contextmanager, nullcontext


import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)

from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from accelerate import Accelerator
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from .utils import *
from models.import_utils import is_peft_available, is_wandb_available
from models import PreTrainedModelWrapper

from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

from copy import deepcopy

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed



class SFTTrainer(transformers.Trainer):
    def __init__(self, 
            model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            data_collator: Optional[DataCollator] = None,
            args: Optional[Any] = None,
            peft_config: Optional[Dict[str, Any]] = None,
            model_adapter_name: Optional[str] = None,
            is_encoder_decoder: Optional[bool] = None,
    ):
        
        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
            
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model.mistral_model, "enable_input_require_grads"): # for videochat2
                model.mistral_model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if is_encoder_decoder is not None:
            warnings.warn(
                "You passed `is_encoder_decoder` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.is_encoder_decoder = is_encoder_decoder
        if model is not None:
            try:
                self.is_encoder_decoder = model.config.is_encoder_decoder
            except:
                self.is_encoder_decoder = False
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder to the DPOTrainer/DPOConfig."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if model_adapter_name is not None:
            warnings.warn(
                "You passed `model_adapter_name` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_adapter_name = model_adapter_name
        self.model_adapter_name = args.model_adapter_name
        
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            data_collator=data_collator,)
        
        self.tokenizer = tokenizer
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        self.generate_during_eval = args.generate_during_eval
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.label_names = ['labels']

        self.args.dataloader_pin_memory = False
        

    def get_batch_generation(self, inputs: Dict[str, torch.Tensor]) -> Tuple[str, str]:
        if self.args.running_model_name in ['vlm-rlaif']:
            model_output = self.model.generate(
                input_ids=inputs['prompt_input_ids'],
                images=inputs['images'],
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.args.response_len,
                temperature=self.args.temperature,
            )
        model_output = pad_to_length(model_output, self.args.response_len, self.tokenizer.pad_token_id)
        model_output[model_output == IMAGE_TOKEN_INDEX] = self.tokenizer.eos_token_id
        model_output[model_output == IGNORE_INDEX] = self.tokenizer.eos_token_id
        model_output_decoded = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

        return model_output_decoded

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = torch.utils.data.Subset(dataloader.dataset, range(self.args.eval_batch_size))
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)
            with torch.no_grad():
                policy_output_decoded = self.get_batch_generation(random_batch)

            random_batch['prompt_input_ids'][random_batch['prompt_input_ids'] == IMAGE_TOKEN_INDEX] = self.tokenizer.eos_token_id
            self.log(
                {
                    "generation_log": wandb.Table(
                        columns=["Step", "Question", "Model Predction"],
                        rows=[
                            [self.state.global_step, self.tokenizer.decode(prompt, skip_special_tokens=True), pol]
                            for prompt, pol in zip(
                                random_batch["prompt_input_ids"], policy_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
            """
            Log `logs` on the various objects watching training, including stored metrics.

            Args:
                logs (`Dict[str, float]`):
                    The values to log.
            """
            # logs either has 'loss' or 'eval_loss'
            train_eval = "train" if "loss" in logs else "eval"
            # Add averaged stored metrics to logs
            for key, metrics in self._stored_metrics[train_eval].items():
                logs[key] = torch.tensor(metrics).mean().item()
            del self._stored_metrics[train_eval]

            if "generation_log" in logs.keys():
                self.state.log_history.append(logs["generation_log"])
                del logs["generation_log"]
            return super().log(logs)

    def get_loss_and_metrics(self, model, batch, train_eval: Literal["train", "eval"] = "train"):
        """
        Overriding built-in get_loss_and_metrics to store metrics for each batch.
        Compute the loss on the given model and batch.

        Subclass and override for custom behavior.
        """

        metrics = {}

        # Compute loss
        if self.args.running_model_name == 'vlm-rlaif':
            outputs = model(input_ids=batch["input_ids"], images=batch["images"], labels=batch["labels"], attention_mask=batch["attention_mask"])
            # calculate all inputes on cpu
            # outputs = model(input_ids=batch["input_ids"].cpu(), pixel_values_videos=batch["images"].cpu(), labels=batch["labels"].cpu(), attention_mask=batch["attention_mask"].cpu())
        try:
            loss = outputs.loss
        except:
            loss = outputs['loss']

        # force log the metrics
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}loss"] = loss.detach().mean().cpu()

        return loss, metrics

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
            for key, value in metrics.items():
                self._stored_metrics[train_eval][key].append(value)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overriding built-in compute_loss to store metrics for each batch.
        Computes the loss of the given model on the given inputs.

        Subclass and override for custom behavior.
        """

        # Compute loss
        loss, metrics = self.get_loss_and_metrics(model, inputs, train_eval="train")
        loss = loss.to(self.args.device)

        # Store metrics
        self.store_metrics(metrics, train_eval="train")

        return (loss, metrics) if return_outputs else loss
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool = True,
        ignore_keys: Optional[List[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_loss_and_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # return (loss.detach(), logits, labels)
        return (loss.detach(), None, None)