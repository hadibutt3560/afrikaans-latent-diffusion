import re
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

import CONSTANTS as CONSTANTS


def set_autoencoder_train_mode(model, backbone_eval_mode=False):
    model.train()
    if backbone_eval_mode:
        model.get_encoder().eval()
        model.get_decoder().eval()

from latent_models.bart_latent_model import BARTForConditionalGenerationLatent
from latent_models.t5_latent_model import (
    MT5ForConditionalGenerationLatent,
    T5ForConditionalGenerationLatent,
)



def get_latent_model(args):
    if 'bart' in args.enc_dec_model:
        config = AutoConfig.from_pretrained(args.enc_dec_model)
        lm = BARTForConditionalGenerationLatent.from_pretrained(
            args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            args.enc_dec_model)
    elif 't5' in args.enc_dec_model:
        if 'mt5' in args.enc_dec_model:
            config = AutoConfig.from_pretrained(args.enc_dec_model)
            lm = MT5ForConditionalGenerationLatent.from_pretrained(
                args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                args.enc_dec_model)
        else:
            config = AutoConfig.from_pretrained(args.enc_dec_model)
            lm = T5ForConditionalGenerationLatent.from_pretrained(
                args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                args.enc_dec_model)
    else:
        print("Unsupported model")
        raise NotImplementedError
    
    if args.lm_mode == 'ft':
        for (param_name, param) in lm.named_parameters():
            param.requires_grad = True
    elif args.lm_mode in {'freeze', 'decoder_last'}:
        for (param_name, param) in lm.named_parameters():
            if re.fullmatch(".*perceiver.*", param_name):
                param.requires_grad = True
                print(f"Trainable: {param_name}")
            else:
                param.requires_grad = False
        if args.lm_mode == 'decoder_last':
            if not hasattr(lm.decoder, 'block'):
                raise ValueError('decoder_last currently supports T5/mT5 only')
            for param in lm.decoder.block[-1].parameters():
                param.requires_grad = True
            for param in lm.decoder.final_layer_norm.parameters():
                param.requires_grad = True
    else:
        raise NotImplementedError
        


        
    return lm, tokenizer, config
