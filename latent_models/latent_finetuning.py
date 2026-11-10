import math
import copy
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
from contextlib import nullcontext
import json
import shutil
from difflib import SequenceMatcher
import re

import torch
from torch import nn, einsum
import torch.nn.functional as F

import timeit

from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm

from transformers import get_scheduler

from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb

import CONSTANTS as CONSTANTS
import diffusion.optimizer as optimizer
import dataset_utils.text_dataset as text_dataset
from utils.torch_utils import compute_grad_norm
import utils.file_utils as file_utils
from evaluation import evaluation

from latent_models.latent_utils import get_latent_model, set_autoencoder_train_mode


generate_kwargs = {
    'beam': 
    {'max_length':64, 'min_length':5, 'do_sample':False, 'num_beams':4, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2},
}

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def set_seeds(seed):
    file_utils.set_seed(seed)

# trainer class

class Trainer(object):
    def __init__(
        self,
        args,
        dataset_name,
        *,
        train_batch_size = 16,
        eval_batch_size = 64,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        num_samples = None,
        eval_every = 1000,
        results_folder = './results',
        mixed_precision = 'no',
        seed=43,
    ):
        super().__init__()


        set_seeds(seed)

        self.args = args
        evaluation.set_metric_policy(
            strict=getattr(args, "strict_evaluation", False),
            skip_expensive=getattr(args, "skip_expensive_metrics", False),
        )

        self.best_val_metric = 0
        self.num_samples = num_samples

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if args.output_dir is None:
            args.output_dir = file_utils.get_output_dir(args)
        results_folder = str(file_utils.ensure_output_dir(args.output_dir))

        tracking_enabled = os.environ.get("WANDB_MODE", "").lower() != "disabled"
        self.accelerator = Accelerator(
            mixed_precision = mixed_precision,
            log_with='wandb' if tracking_enabled else None,
            kwargs_handlers=[ddp_kwargs]
        )
        self.tracking_enabled = tracking_enabled
        self.num_devices = self.accelerator.num_processes
        args.num_devices = self.num_devices

        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            if tracking_enabled:
                if args.wandb_name:
                    self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
                else:
                    self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        self.enc_dec_model = args.enc_dec_model

        self.lm, self.tokenizer, config = get_latent_model(args)
        num_trainable_params = sum(p.numel() for p in self.lm.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            self.accelerator.print(f'num trainable params: {num_trainable_params}')

        self.eval_every = eval_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        self.dataset = text_dataset.get_dataset(
            dataset_name,
            dataset_dir=getattr(args, "dataset_dir", None),
        )

        subset_size = getattr(args, 'train_subset_size', 0)
        if subset_size:
            self.dataset['train'] = self.dataset['train'].shuffle(seed=seed).select(
                range(min(subset_size, len(self.dataset['train'])))
            )
        if getattr(args, 'overfit_subset', False):
            if not subset_size:
                raise ValueError('overfit_subset requires train_subset_size')
            self.dataset['valid'] = self.dataset['train']
        elif getattr(args, 'validation_shuffle', False):
            self.dataset['valid'] = self.dataset['valid'].shuffle(seed=seed)

        if args.eval or getattr(args, "eval_test", False):
            self.dataset['train'] = self.dataset['train'].select(range(min(1000, len(self.dataset['train']))))
        self.dataloader = text_dataset.get_dataloader(args, self.dataset['train'], config, self.tokenizer, args.max_seq_len, context_tokenizer=self.tokenizer, is_train=True)
        self.val_dataloader = text_dataset.get_dataloader(args, self.dataset['valid'], config, self.tokenizer, args.max_seq_len, shuffle=False, context_tokenizer=self.tokenizer, is_train=False)
        self.test_dataloader = text_dataset.get_dataloader(args, self.dataset['test'], config, self.tokenizer, args.max_seq_len, shuffle=False, context_tokenizer=self.tokenizer, is_train=False)
        self.max_seq_len = args.max_seq_len

        # optimizer

        # check that the new latents are added to the optimizer
        if args.optimizer == 'adamw':
            self.opt = optimizer.get_adamw_optimizer(self.lm.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)
        elif args.optimizer == 'lion':
            self.opt = optimizer.get_lion_optimizer(self.lm.parameters(), lr = train_lr, weight_decay=adam_weight_decay)
        else:
            raise ValueError(f'invalid optimizer {args.optimizer}')

        # scheduler

        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps*self.num_devices,
            num_training_steps=train_num_steps*self.num_devices,
        )

        # for logging results in a folder periodically

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.lm, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader, self.test_dataloader = self.accelerator.prepare(self.lm, self.opt, self.dataloader, lr_scheduler, self.val_dataloader, self.test_dataloader)
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)

        args.gradient_accumulation_steps = self.gradient_accumulate_every
        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.lm.parameters())
            trainable_params = sum(p.numel() for p in self.lm.parameters() if p.requires_grad)
            file_utils.write_run_metadata(
                self.results_folder,
                args,
                dataset_counts={split: len(values) for split, values in self.dataset.items()},
                model_counts={"total": total_params, "trainable": trainable_params},
                extra={"run_type": "autoencoder"},
            )
            self.accelerator.print(
                "Batching: "
                f"{self.train_batch_size} per GPU x {self.gradient_accumulate_every} accumulation "
                f"x {self.num_devices} GPU(s) = "
                f"{self.train_batch_size * self.gradient_accumulate_every * self.num_devices} global"
            )

    def save(self):
        if not self.accelerator.is_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.lm),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'scheduler': self.lr_scheduler.state_dict(),
        }

        checkpoint_path = self.results_folder / 'model.pt'
        temporary_path = checkpoint_path.with_suffix('.pt.tmp')
        torch.save(data, str(temporary_path))
        os.replace(temporary_path, checkpoint_path)

    def log_local_metrics(self, logs):
        if not self.accelerator.is_main_process:
            return
        metrics_path = self.results_folder / 'training_metrics.jsonl'
        serializable = {
            key: value.item() if hasattr(value, 'item') else value
            for key, value in logs.items()
        }
        with open(metrics_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(serializable, sort_keys=True) + '\n')

    def load(self, file_path=None, resume_training=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.lm)
        model.load_state_dict(data['model'])

        self.step = data['step']
        if resume_training:
            self.opt.load_state_dict(data['opt'])
            if exists(self.accelerator.scaler) and exists(data.get('scaler')):
                self.accelerator.scaler.load_state_dict(data['scaler'])
            if 'scheduler' in data:
                self.lr_scheduler.load_state_dict(data['scheduler'])
            else:
                # Backwards compatibility for old checkpoints.
                for _ in range(self.step):
                    self.lr_scheduler.step()

    
    @torch.no_grad()
    def validation(self, split="valid"):
        if split not in {"valid", "test"}:
            raise ValueError(f"Unknown reconstruction split: {split}")
        evaluation_dataloader = (
            self.val_dataloader if split == "valid" else self.test_dataloader
        )
        self.lm.eval()
        pred_text = {k:[] for k,_ in generate_kwargs.items()}    
        bart_text = {k:[] for k,_ in generate_kwargs.items()}    
        ref_text = []
        total_nll = 0.0
        total_tokens = 0
        correct_tokens = 0
        accelerator = self.accelerator
        device = self.accelerator.device
        for batch in tqdm(evaluation_dataloader):
            data = {k:v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                if self.num_devices > 1:
                    metric_encoder_outputs = self.lm.module.get_encoder()(
                        input_ids=data['input_ids'], attention_mask=data['attention_mask']
                    )
                    metric_encoder_outputs = self.lm.module.encoder_output_to_decoder_input(
                        metric_encoder_outputs, data['attention_mask']
                    )
                else:
                    metric_encoder_outputs = self.lm.get_encoder()(
                        input_ids=data['input_ids'], attention_mask=data['attention_mask']
                    )
                    metric_encoder_outputs = self.lm.encoder_output_to_decoder_input(
                        metric_encoder_outputs, data['attention_mask']
                    )
                metric_outputs = self.lm(
                    labels=data['labels'], encoder_outputs=metric_encoder_outputs
                )
                label_mask = data['labels'].ne(-100)
                token_count = int(label_mask.sum().item())
                total_nll += float(metric_outputs.loss.item()) * token_count
                total_tokens += token_count
                correct_tokens += int(
                    (metric_outputs.logits.argmax(dim=-1).eq(data['labels']) & label_mask)
                    .sum()
                    .item()
                )
            for strategy in generate_kwargs.keys():
                gen_kwargs = generate_kwargs[strategy]
                gen_kwargs['max_length'] = self.max_seq_len
                # Compute generated language
                if self.num_devices > 1:
                    encoder_outputs = self.lm.module.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                    encoder_outputs = self.lm.module.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])
                    sample_ids = self.lm.module.generate(encoder_outputs=encoder_outputs, **gen_kwargs)
                else:
                    encoder_outputs = self.lm.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                    encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])
                    sample_ids = self.lm.generate(encoder_outputs=encoder_outputs, **gen_kwargs)
                # Pad sample_ids to max_seq_len
                sample_ids = F.pad(sample_ids, (0, self.max_seq_len - sample_ids.shape[-1]), value=self.tokenizer.pad_token_id)
                gathered_sample_ids = accelerator.gather(sample_ids).to('cpu')
                texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in gathered_sample_ids]
                pred_text[strategy].extend(texts_list)

                # Compute BART language
                if self.num_devices > 1:
                    sample_ids2 = self.lm.module.generate(input_ids = data['input_ids'], attention_mask = data['attention_mask'], **gen_kwargs)
                else:
                    sample_ids2 = self.lm.generate(input_ids = data['input_ids'], attention_mask = data['attention_mask'], **gen_kwargs)
                sample_ids2 = F.pad(sample_ids2, (0, self.max_seq_len - sample_ids2.shape[-1]), value=self.tokenizer.pad_token_id)
                gathered_sample_ids2 = accelerator.gather(sample_ids2).to('cpu')
                texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in gathered_sample_ids2]
                bart_text[strategy].extend(texts_list)

            # Store reference language
            gathered_input_ids = accelerator.gather(data['input_ids']).to('cpu')
            texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in gathered_input_ids]
            ref_text.extend(texts_list)
            if len(ref_text) >= getattr(self.args, "validation_max_examples", 1000):
                break

        validation_stats = torch.tensor(
            [total_nll, float(total_tokens), float(correct_tokens)],
            dtype=torch.float64,
            device=device,
        )
        validation_stats = accelerator.gather(validation_stats).reshape(-1, 3).sum(dim=0)
        total_nll, total_tokens, correct_tokens = validation_stats.tolist()

        if not self.accelerator.is_main_process:
            return
        # Compute metrics
        maximum = getattr(self.args, "validation_max_examples", 1000)
        ref_text = ref_text[:maximum]
        for strategy in generate_kwargs:
            pred_text[strategy] = pred_text[strategy][:len(ref_text)]
            bart_text[strategy] = bart_text[strategy][:len(ref_text)]
        metrics = {
            f"autoencoder/{split}_cross_entropy": total_nll / total_tokens if total_tokens else float("nan"),
            f"autoencoder/{split}_token_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
        }
        for strategy in generate_kwargs.keys():
            metrics[f'autoencoder/{strategy}/exact_reconstruction'] = (
                sum(pred == ref for pred, ref in zip(pred_text[strategy], ref_text)) / len(ref_text)
                if ref_text else 0.0
            )
            pairs = list(zip(pred_text[strategy], ref_text))
            metrics[f'autoencoder/{strategy}/character_similarity'] = (
                sum(SequenceMatcher(None, ref, pred, autojunk=False).ratio()
                    for pred, ref in pairs) / len(pairs) if pairs else 0.0
            )
            repeated = []
            expected_numbers, matched_numbers = 0, 0
            for pred, ref in pairs:
                words = pred.split()
                trigrams = list(zip(words, words[1:], words[2:]))
                repeated.append(1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0)
                expected = Counter(re.findall(r'\d+(?:[.,]\d+)*', ref))
                actual = Counter(re.findall(r'\d+(?:[.,]\d+)*', pred))
                expected_numbers += sum(expected.values())
                matched_numbers += sum((expected & actual).values())
            metrics[f'autoencoder/{strategy}/repeated_trigram_fraction'] = (
                sum(repeated) / len(repeated) if repeated else 0.0
            )
            metrics[f'autoencoder/{strategy}/number_recall'] = (
                matched_numbers / expected_numbers if expected_numbers else None
            )
            if not getattr(self.args, "skip_expensive_metrics", False):
                metrics[f'autoencoder/{strategy}/bleu'] = evaluation.compute_bleu(pred_text[strategy], ref_text)
                metrics[f'bart/{strategy}/bleu'] = evaluation.compute_bleu(bart_text[strategy], ref_text)
                metrics[f'autoencoder/{strategy}/perplexity'] = evaluation.compute_perplexity(pred_text[strategy])
                metrics[f'bart/{strategy}/perplexity'] = evaluation.compute_perplexity(bart_text[strategy])
                for key, value in evaluation.compute_rouge(pred_text[strategy], ref_text).items():
                    metrics[f'autoencoder/{strategy}/{key}'] = value
                for key, value in evaluation.compute_rouge(bart_text[strategy], ref_text).items():
                    metrics[f'bart/{strategy}/{key}'] = value
        if not getattr(self.args, "skip_expensive_metrics", False):
            metrics['reference/perplexity'] = evaluation.compute_perplexity(ref_text)
         

        accelerator.log(metrics, self.step)

        # Log samples
        # reference | strategy0/autoencoder | strategy0/bart | strategy1/autoencoder | strategy1/bart | ...
        columns = ['reference'] + [f'{strategy}/autoencoder' for strategy in generate_kwargs.keys()] + [f'{strategy}/bart' for strategy in generate_kwargs.keys()]
        data = []
        for i in range(len(ref_text)):
            row = [ref_text[i]]
            for strategy in generate_kwargs.keys():
                row.append(pred_text[strategy][i])
            
            for strategy in generate_kwargs.keys():
                row.append(bart_text[strategy][i])
            data.append(row)
        if self.tracking_enabled:
            table = wandb.Table(columns=columns, data=data)
            accelerator.log({f"Samples": table}, self.step)

        metrics_path = self.results_folder / f"autoencoder-{split}-step-{self.step}.json"
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
        examples_path = self.results_folder / f"reconstruction-{split}-examples-step-{self.step}.jsonl"
        with open(examples_path, "w", encoding="utf-8") as handle:
            for index, reference in enumerate(ref_text):
                row = {"reference": reference}
                for strategy in generate_kwargs:
                    row[f"{strategy}_autoencoder"] = pred_text[strategy][index]
                    row[f"{strategy}_bart"] = bart_text[strategy][index]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return metrics

    def safe_validation(self):
        try:
            metrics = self.validation()
            if self.accelerator.is_main_process and metrics:
                score = metrics['autoencoder/valid_cross_entropy']
                best_dir = self.results_folder / 'best'
                score_path = best_dir / 'selection.json'
                previous = float('inf')
                if score_path.exists():
                    previous = json.loads(score_path.read_text())['cross_entropy']
                if math.isfinite(score) and score < previous:
                    best_dir.mkdir(exist_ok=True)
                    shutil.copy2(self.results_folder / 'model.pt', best_dir / 'model.pt.tmp')
                    os.replace(best_dir / 'model.pt.tmp', best_dir / 'model.pt')
                    shutil.copy2(self.results_folder / 'args.json', best_dir / 'args.json')
                    score_path.write_text(json.dumps({'step': self.step, 'cross_entropy': score}))
        except Exception as exc:
            if getattr(self.args, "strict_evaluation", False):
                raise
            if self.accelerator.is_main_process:
                self.accelerator.print(f'Validation failed at step {self.step}: {exc}')

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        self.set_training_mode()
        if self.args.lm_mode in {'freeze', 'decoder_last'}:
            encoder_context = torch.no_grad()
        else:
            encoder_context = nullcontext()

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            training_started = timeit.default_timer()

            while self.step < self.train_num_steps:
                budget = getattr(self.args, 'max_train_seconds', 0)
                if budget > 0 and timeit.default_timer() - training_started >= budget:
                    accelerator.print('Training wall-time budget reached; saving at a step boundary.')
                    break

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = {k:v.to(device) for k,v in next(self.data_iter).items()}

                    with accelerator.autocast():
                        with encoder_context:
                            if self.num_devices > 1:
                                encoder_outputs = self.lm.module.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                            else:
                                encoder_outputs = self.lm.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                        if self.num_devices > 1:
                            encoder_outputs = self.lm.module.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])
                        else:
                            encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])

                        raw_loss = self.lm(labels=data['labels'], encoder_outputs=encoder_outputs).loss
                        if not torch.isfinite(raw_loss):
                            raise FloatingPointError(f'Non-finite loss at step {self.step}')
                        loss = raw_loss / self.gradient_accumulate_every
                    total_loss += raw_loss.item() / self.gradient_accumulate_every
                    self.accelerator.backward(loss)


                accelerator.wait_for_everyone()

                grad_norm = compute_grad_norm(self.lm.parameters())

                accelerator.clip_grad_norm_(self.lm.parameters(), self.args.clip_grad_norm)
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                # Log to WandB
                if self.step % 50 == 0:
                    self.lm.eval()
                    with torch.no_grad():
                        total_val_loss = 0.
                        total_lm_val_loss = 0.
                        data = {k:v.to(device) for k,v in next(self.val_iter).items()}

                        if self.num_devices > 1:
                            encoder_outputs = self.lm.module.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                            encoder_outputs = self.lm.module.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])
                        else:
                            encoder_outputs = self.lm.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                            encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])
                        loss = self.lm(labels=data['labels'], encoder_outputs=encoder_outputs).loss                      
                        if self.args.lm_mode == 'freeze':
                            total_lm_val_loss += self.lm(input_ids = data['input_ids'], attention_mask = data['attention_mask'], labels=data['labels']).loss.item()
                        total_val_loss += loss.item()

                        logs = {"train/loss": total_loss, "val/loss": total_val_loss, "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step, "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader), "samples": self.step*self.train_batch_size*self.gradient_accumulate_every*self.num_devices}
                        if self.args.lm_mode == 'freeze':
                            logs["val/lm_loss"] = total_lm_val_loss
                        pbar.set_postfix(**logs)
                            
                    self.set_training_mode()
                else:
                    logs = {"train/loss": total_loss, "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step, "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader), "samples": self.step*self.train_batch_size*self.gradient_accumulate_every*self.num_devices}

                if accelerator.is_main_process:
                    accelerator.log(logs, step=self.step)
                    self.log_local_metrics(logs)

                if self.step % self.eval_every == 0:
                    self.save()
                    accelerator.wait_for_everyone()
                    self.safe_validation()
                    accelerator.wait_for_everyone()
                    self.set_training_mode()

                pbar.update(1)
        self.save()
        accelerator.wait_for_everyone()
        self.safe_validation()

        accelerator.print('training complete')

    def set_training_mode(self):
        set_autoencoder_train_mode(
            self.accelerator.unwrap_model(self.lm),
            getattr(self.args, 'backbone_eval_mode', False),
        )
