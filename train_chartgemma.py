from datasets import load_dataset
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
)
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.utils.data import Dataset
from typing import Any, Dict
import random
from PIL import Image
from io import BytesIO
import lightning as L
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np
import torch.nn.functional as F


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


class ChartGemmaDataset(Dataset):
    """
    PyTorch Dataset for ChartGemma. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and ground truth data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
    ):
        super().__init__()

        self.split = split
        self.dataset = load_dataset(dataset_name_or_path, split=split)
        self.dataset_length = len(self.dataset)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # inputs
        image = Image.open(BytesIO(sample["image"])).convert("RGB")
        target_sequence = sample["output"]
        input_sequence = sample["input"]
        return image, input_sequence, target_sequence


def train_collate_fn(examples):
    images = []
    rot_images = []
    input_texts = []
    outputs_texts = []

    for example in examples:
        image, input_text, output_text = example
        images.append(image)

        # Later, 90, 180, 270 randomly?
        rot_images.append(image.rotate(-90, expand=True))
        input_texts.append(input_text)
        outputs_texts.append(output_text)

    # Change the MX LENGTH depending on the task.
    MAX_LENGTH = 128
    inputs = processor(
        text=input_texts,
        images=images,
        suffix=outputs_texts,
        return_tensors="pt",
        padding=True,
        truncation="only_second",
        max_length=MAX_LENGTH,
        # tokenize_newline_separately=False,
    )

    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    labels = inputs["labels"]

    inputs = processor(
        text=input_texts,
        images=rot_images,
        suffix=outputs_texts,
        return_tensors="pt",
        padding=True,
        truncation="only_second",
        max_length=MAX_LENGTH,
        # tokenize_newline_separately=False,
    )
    rot_pixel_values = inputs["pixel_values"]

    return (
        input_ids,
        token_type_ids,
        attention_mask,
        pixel_values,
        rot_pixel_values,
        labels,
    )


def eval_collate_fn(examples):
    # we only feed the prompt to the model
    images = []
    texts = []
    answers = []
    for example in examples:
        image, text, answer = example
        images.append(image)
        texts.append(text)
        answers.append(answer)

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        tokenize_newline_separately=False,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    return input_ids, attention_mask, pixel_values, answers


class ChartGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model, train_dataset, val_dataset):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def training_step(self, batch, batch_idx):

        (
            input_ids,
            token_type_ids,
            attention_mask,
            pixel_values,
            rot_pixel_values,
            labels,
        ) = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = outputs.loss

        # for consistency loss
        rot_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=rot_pixel_values,
            labels=labels,
        )

        logit_soft = F.log_softmax(outputs.logits, dim=-1)
        rot_logit_soft = F.log_softmax(rot_outputs.logits, dim=-1)
        kl_loss = F.kl_div(logit_soft, rot_logit_soft, reduction="batchmean")

        # loss weight is default 1:1
        loss = loss + kl_loss

        import pdb

        pdb.set_trace()
        self.log("train_loss", loss)

        return loss

    def compute_metric(self, gt, pred):
        try:
            gt = float(gt)
            pred = float(pred)
            return abs(gt - pred) / abs(gt) <= 0.05
        except:
            return str(gt).lower() == str(pred).lower()

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=128,
        )
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(
            generated_ids[:, input_ids.size(1) :], skip_special_tokens=True
        )

        scores = []
        for pred, answer in zip(predictions, answers):
            # pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            correct = self.compute_metric(answer, pred.strip())
            if correct:
                scores.append(1)
            else:
                scores.append(0)

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")

        self.log("val_relaxed_accuracy", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            train_dataset,
            collate_fn=train_collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            val_dataset,
            collate_fn=eval_collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )


USE_LORA = True
USE_QLORA = False

if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "ahmed-masry/chartgemma",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
elif USE_LORA:
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "ahmed-masry/chartgemma",
        torch_dtype=torch.float16,
    )
else:
    # for full fine-tuning, we can speed up the model using Flash Attention
    # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "ahmed-masry/chartgemma",
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

train_dataset = ChartGemmaDataset("ahmed-masry/ChartGemma", split="train")
val_dataset = ChartGemmaDataset("ahmed-masry/ChartGemma", split="train")

config = {
    "max_epochs": 2,
    # "val_check_interval": 0.2, # how many times we want to validate during an epoch
    "check_val_every_n_epoch": None,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 8,
    "accumulate_grad_batches": 1,
    "lr": 1e-4,
    "batch_size": 1,
    # "seed":2022,
    "num_nodes": 1,
    "warmup_steps": 50,
    "result_path": "./result",
    "verbose": True,
}

model_module = ChartGemmaModelPLModule(
    config, processor, model, train_dataset, val_dataset
)

from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="chartgemma", name="KL_base")

trainer = L.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    precision="16-mixed",
    num_sanity_val_steps=0,
    logger=wandb_logger,
)

trainer.fit(model_module)

model_module.model.save_pretrained("trained_model")
model_module.processor.save_pretrained("trained_model")
