# ASR Ensemble Function for HW4P2
def ensemble_from_wandb(run_ids: list, temperature: float = 1.0, beam_width: int = 5, length_penalty: float = 0.8):
    """
    Load multiple ASR models from wandb and create an ensemble for improved CER performance.
    Adapted from HW3P2 ensemble approach for ASR models.

    Args:
        run_ids: List of wandb run IDs to load models from
        temperature: Temperature for softmax scaling
        beam_width: Beam width for beam search decoding
        length_penalty: Length penalty for beam search

    Returns:
        ensemble_model: Callable that takes (audio_features, feature_lengths) and returns predictions
    """
    import wandb
    import torch
    import re
    import yaml
    from hw4lib.model.transformers import EncoderDecoderTransformer
    from hw4lib.decoding.sequence_generator import SequenceGenerator
    from hw4lib.data.tokenizer import H4Tokenizer

    models = []
    configs = []
    tokenizers = []

    print(f"Loading {len(run_ids)} models for ensemble...")

    for i, run_id in enumerate(run_ids):
        print(f"\n=== Loading Model {i+1}/{len(run_ids)} ===")
        print(f"Run ID: {run_id}")

        # Initialize wandb run
        run = wandb.init(
            project="HW4P2",
            id=run_id,
            resume="must",
            mode="disabled"  # Don't log during ensemble loading
        )

        try:
            # Download model files
            print("Downloading model files...")
            model_file = wandb.restore('best_model.pth', run_path=f"idlf25/HW4P2/{run_id}")
            config_file = wandb.restore('config.yaml', run_path=f"idlf25/HW4P2/{run_id}")
            arch_file = wandb.restore('model_arch.txt', run_path=f"idlf25/HW4P2/{run_id}")

            # Load config
            with open(config_file.name, 'r') as f:
                config = yaml.safe_load(f)
            configs.append(config)

            # Create tokenizer
            token_type = config['tokenization']['token_type']
            tokenizer_path = config['tokenization']['token_map'][token_type]
            tokenizer = H4Tokenizer(tokenizer_path)
            tokenizers.append(tokenizer)

            print(f"Config loaded: {config['Name']}")
            print(f"Tokenizer: {token_type} ({tokenizer.get_vocab_size()} tokens)")

            # Parse model architecture from model_arch.txt
            with open(arch_file.name, 'r') as f:
                arch_content = f.read()

            # Extract parameter count
            param_match = re.search(r'Total params: ([\d,]+)', arch_content)
            if param_match:
                param_count = param_match.group(1)
                print(f"Model parameters: {param_count}")

            # Create model with config
            model_config = config['model'].copy()
            model_config['num_classes'] = tokenizer.get_vocab_size()
            model_config['max_len'] = 1024  # Default max length

            print(f"Creating model with d_model={model_config['d_model']}, "
                  f"enc_layers={model_config['num_encoder_layers']}, "
                  f"dec_layers={model_config['num_decoder_layers']}")

            # Initialize model
            model = EncoderDecoderTransformer(**model_config)

            # Load model weights
            checkpoint = torch.load(model_file.name, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")

            if torch.cuda.is_available():
                model = model.cuda()
                print("Model moved to GPU")

            models.append(model)

        except Exception as e:
            print(f"Error loading model {run_id}: {e}")
            continue
        finally:
            wandb.finish()

    print(f"\nEnsemble loaded: {len(models)} models ready")

    def ensemble_predict(audio_features, feature_lengths, return_logits=False):
        """
        Ensemble prediction function for ASR.

        Args:
            audio_features: Input audio features (batch_size, seq_len, feat_dim)
            feature_lengths: Feature sequence lengths (batch_size,)
            return_logits: Whether to return raw logits or decoded text

        Returns:
            If return_logits=True: averaged logits for further processing
            If return_logits=False: list of decoded transcriptions
        """
        if len(models) == 0:
            raise ValueError("No models loaded in ensemble")

        device = next(models[0].parameters()).device
        if audio_features.device != device:
            audio_features = audio_features.to(device)
            feature_lengths = feature_lengths.to(device)

        # Encode with all models and average encoder outputs
        encoder_outputs = []
        pad_masks = []

        with torch.no_grad():
            for model in models:
                enc_out, pad_mask, _, _ = model.encode(audio_features, feature_lengths)
                encoder_outputs.append(enc_out)
                pad_masks.append(pad_mask)

        # Average encoder outputs (weighted equally)
        avg_encoder_output = torch.stack(encoder_outputs).mean(dim=0)
        # Use pad mask from first model (should be identical)
        pad_mask = pad_masks[0]

        # Use first model's tokenizer for decoding
        tokenizer = tokenizers[0]

        if return_logits:
            # For logit-level ensemble, decode with averaged encoder output
            generator = SequenceGenerator(models[0], tokenizer)
            return generator.greedy_search(avg_encoder_output, pad_mask, max_len=256)
        else:
            # For prediction-level ensemble, decode with beam search
            generator = SequenceGenerator(models[0], tokenizer)

            # Generate with averaged encoder output
            predictions = generator.beam_search(
                encoder_output=avg_encoder_output,
                pad_mask_src=pad_mask,
                beam_width=beam_width,
                max_len=256,
                temperature=temperature,
                length_penalty=length_penalty
            )

            return predictions

    return ensemble_predict