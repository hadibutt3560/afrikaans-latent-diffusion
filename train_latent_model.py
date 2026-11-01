import argparse
import json
import os


def build_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate the LD4LG language autoencoder")
    parser.add_argument("--dataset_name", type=str, default="roc")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Optional directory containing train/valid/test JSONL files.",
    )
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--enc_dec_model", type=str, default="facebook/bart-base")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Per-device micro-batch size (not the global batch size).",
    )
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_encoder_latents", type=int, default=32)
    parser.add_argument("--num_decoder_latents", type=int, default=32)
    parser.add_argument("--dim_ae", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--l2_normalize_latents", action="store_true")
    # Backwards-compatible alias for the typo in the original upstream shell script.
    parser.add_argument(
        "--l2_normalize_latent",
        dest="l2_normalize_latents",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_latent_models")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_steps", type=int, default=50000)
    parser.add_argument("--max_train_seconds", type=float, default=0, help="Stop at an optimizer-step boundary and save after this elapsed training budget; zero disables.")
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--validation_max_examples", type=int, default=1000)
    parser.add_argument("--skip_expensive_metrics", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode. BF16 requires a supported GPU.",
    )
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument(
        "--lm_mode",
        type=str,
        default="freeze",
        choices=["freeze", "decoder_last", "ft"],
        help="Whether the pretrained encoder-decoder is frozen or fine-tuned.",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--init_dir", default=None, help="Load model weights only; start a fresh optimizer and step counter.")
    parser.add_argument("--train_subset_size", type=int, default=0)
    parser.add_argument("--overfit_subset", action="store_true", help="Diagnostic only: evaluate on the selected training subset.")
    parser.add_argument("--validation_shuffle", action="store_true")
    parser.add_argument("--fail_on_validation_error", action="store_true")
    parser.add_argument("--backbone_eval_mode", action="store_true", help="Disable backbone training-time dropout without disabling parameter gradients; diagnostic ablation.")
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument(
        "--resume_to_step",
        type=int,
        default=None,
        help="Override the saved training horizon when resuming.",
    )
    parser.add_argument(
        "--allow_missing_metrics",
        action="store_true",
        help="Allow unavailable metrics in explicit evaluation. Do not use for final results.",
    )
    return parser


def merge_saved_args(args):
    if not (args.eval or args.eval_test or args.resume_training):
        return args
    if not args.resume_dir:
        raise ValueError(
            "--resume_dir is required with --eval, --eval_test, or --resume_training"
        )
    with open(os.path.join(args.resume_dir, "args.json"), "rt", encoding="utf-8") as handle:
        saved_args = json.load(handle)
    requested_validation_shuffle = args.validation_shuffle
    heldout = {
        "wandb_name", "output_dir", "resume_dir", "eval", "eval_test", "resume_training",
        "allow_missing_metrics", "skip_expensive_metrics", "validation_max_examples",
        "resume_to_step", "init_dir", "max_train_seconds",
    }
    for key, value in saved_args.items():
        if key not in heldout:
            setattr(args, key, value)
    if args.resume_to_step is not None:
        args.num_train_steps = args.resume_to_step
    if requested_validation_shuffle:
        args.validation_shuffle = True
    return args


def main(args):
    from latent_models.latent_finetuning import Trainer
    from utils import file_utils

    if args.init_dir and args.resume_dir:
        raise ValueError('Use either init_dir or resume_dir, not both')
    if args.train_subset_size < 0:
        raise ValueError('train_subset_size must be nonnegative')

    file_utils.set_seed(args.seed, deterministic=args.deterministic)
    args.strict_evaluation = bool(
        ((args.eval or args.eval_test) and not args.allow_missing_metrics)
        or args.fail_on_validation_error
    )

    trainer = Trainer(
        args=args,
        dataset_name=args.dataset_name,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulate_every=args.gradient_accumulation_steps,
        train_lr=args.learning_rate,
        train_num_steps=args.num_train_steps,
        lr_schedule=args.lr_schedule,
        num_warmup_steps=args.lr_warmup_steps,
        adam_betas=(args.adam_beta1, args.adam_beta2),
        adam_weight_decay=args.adam_weight_decay,
        eval_every=args.eval_every,
        results_folder=args.output_dir,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
    )

    if args.resume_dir:
        trainer.load(args.resume_dir, resume_training=args.resume_training)
    elif args.init_dir:
        trainer.load(args.init_dir, resume_training=False)
        trainer.step = 0

    if args.eval or args.eval_test:
        trainer.validation(split="test" if args.eval_test else "valid")
        return
    trainer.train()


if __name__ == "__main__":
    parsed_args = merge_saved_args(build_parser().parse_args())
    if sum([parsed_args.eval, parsed_args.eval_test, parsed_args.resume_training]) > 1:
        raise ValueError(
            "--eval, --eval_test, and --resume_training are mutually exclusive"
        )
    main(parsed_args)
