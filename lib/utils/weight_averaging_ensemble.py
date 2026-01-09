# Weight Averaging Ensemble for Enhanced Performance
import torch
import wandb
import yaml
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
from lib.model.transformers import EncoderDecoderTransformer
from lib.data.tokenizer import H4Tokenizer

def create_weight_averaged_model(
    run_ids: List[str],
    weights: Optional[List[float]] = None,
    save_averaged_model: bool = True,
    model_name: str = "weight_averaged_ensemble"
) -> Tuple[EncoderDecoderTransformer, Dict]:
    """
    Create a weight-averaged ensemble model by averaging model parameters.
    This is often more effective than output averaging for similar architectures.

    Args:
        run_ids: List of wandb run IDs to average
        weights: Optional weights for each model (defaults to equal weighting)
        save_averaged_model: Whether to save the averaged model
        model_name: Name for the averaged model

    Returns:
        averaged_model: The weight-averaged EncoderDecoderTransformer
        metadata: Information about the averaging process
    """
    import os
    import tempfile

    print(f"🤝 Creating weight-averaged ensemble from {len(run_ids)} models")

    if weights is None:
        weights = [1.0 / len(run_ids)] * len(run_ids)
    else:
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

    print(f"📊 Model weights: {[f'{w:.3f}' for w in weights]}")

    models = []
    configs = []

    # Download and load all models
    for i, run_id in enumerate(run_ids):
        print(f"\n📥 Loading model {i+1}/{len(run_ids)}: {run_id}")

        api = wandb.Api()
        run = api.run(f"idlf25/P2/{run_id}")

        # Create temp directory for this model
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files
            model_file = run.file('best_model.pth')
            config_file = run.file('config.yaml')

            model_path = os.path.join(temp_dir, 'model.pth')
            config_path = os.path.join(temp_dir, 'config.yaml')

            model_file.download(root=temp_dir, replace=True)
            config_file.download(root=temp_dir, replace=True)

            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            configs.append(config)

            # Create tokenizer
            token_type = config['tokenization']['token_type']
            token_map = config['tokenization']['token_map']
            tokenizer = H4Tokenizer(token_map, token_type)

            # Create model
            model_config = config['model'].copy()
            model_config['num_classes'] = tokenizer.vocab_size
            model_config['max_len'] = 1024

            model = EncoderDecoderTransformer(**model_config)

            # Load weights
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])

            models.append(model)
            print(f"✅ Loaded: {config['Name']} (epoch {checkpoint.get('epoch', '?')})")

    # Verify all models have the same architecture
    print("\n🔍 Verifying model compatibility...")
    reference_state_dict = models[0].state_dict()

    for i, model in enumerate(models[1:], 1):
        current_state_dict = model.state_dict()

        # Check if keys match
        if set(reference_state_dict.keys()) != set(current_state_dict.keys()):
            raise ValueError(f"Model {i} has incompatible architecture (different keys)")

        # Check if shapes match
        for key in reference_state_dict.keys():
            if reference_state_dict[key].shape != current_state_dict[key].shape:
                raise ValueError(f"Model {i} has incompatible shape for {key}")

    print("✅ All models are compatible for weight averaging")

    # Create averaged state dict
    print("\n🧮 Computing weighted average of parameters...")
    averaged_state_dict = OrderedDict()

    for key in reference_state_dict.keys():
        # Weight average the parameters
        averaged_param = torch.zeros_like(reference_state_dict[key])

        for model, weight in zip(models, weights):
            averaged_param += weight * model.state_dict()[key]

        averaged_state_dict[key] = averaged_param

    # Create new model with averaged weights
    print("🏗️  Creating averaged model...")
    averaged_model = EncoderDecoderTransformer(**model_config)
    averaged_model.load_state_dict(averaged_state_dict)

    # Calculate total parameters
    total_params = sum(p.numel() for p in averaged_model.parameters())
    print(f"📊 Averaged model parameters: {total_params:,}")

    # Prepare metadata
    metadata = {
        'source_run_ids': run_ids,
        'weights': weights,
        'model_configs': configs,
        'total_parameters': total_params,
        'averaging_timestamp': torch.datetime.now().isoformat(),
        'model_name': model_name
    }

    # Save averaged model if requested
    if save_averaged_model:
        os.makedirs('averaged_models', exist_ok=True)

        averaged_model_path = f'averaged_models/{model_name}.pth'
        metadata_path = f'averaged_models/{model_name}_metadata.yaml'

        # Save model
        torch.save({
            'model_state_dict': averaged_state_dict,
            'metadata': metadata,
            'config': configs[0]  # Use first config as reference
        }, averaged_model_path)

        # Save metadata
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, indent=2)

        print(f"💾 Averaged model saved to: {averaged_model_path}")
        print(f"💾 Metadata saved to: {metadata_path}")

        # Upload to wandb if available
        if wandb.run is not None:
            wandb.save(averaged_model_path)
            wandb.save(metadata_path)
            print("☁️  Uploaded to wandb")

    return averaged_model, metadata


def smart_weight_averaging(
    run_ids: List[str],
    validation_loader,
    weighting_strategy: str = 'performance',
    tokenizer: H4Tokenizer = None
) -> Tuple[EncoderDecoderTransformer, Dict]:
    """
    Create a weight-averaged ensemble using smart weighting strategies.

    Args:
        run_ids: List of wandb run IDs
        validation_loader: Validation dataloader for performance measurement
        weighting_strategy: 'equal', 'performance', or 'inverse_loss'
        tokenizer: H4Tokenizer instance

    Returns:
        averaged_model: The weight-averaged model
        metadata: Metadata including individual model performances
    """
    print(f"🧠 Smart weight averaging with strategy: {weighting_strategy}")

    if weighting_strategy == 'equal':
        weights = None  # Will be set to equal in create_weight_averaged_model

    elif weighting_strategy == 'performance':
        print("📈 Evaluating individual model performances...")

        performances = []
        for run_id in run_ids:
            # Load individual model and evaluate
            from ensemble_asr import ensemble_from_wandb
            single_model = ensemble_from_wandb([run_id])

            # Quick evaluation (implement based on your evaluation function)
            cer = evaluate_model_cer(single_model, validation_loader, tokenizer)
            performances.append(1.0 / (cer + 0.01))  # Inverse CER as weight
            print(f"  {run_id}: CER = {cer:.4f}, Weight = {performances[-1]:.3f}")

        # Normalize weights
        total_perf = sum(performances)
        weights = [p / total_perf for p in performances]

    elif weighting_strategy == 'inverse_loss':
        # Weight by inverse of validation loss (if available in wandb)
        weights = get_weights_from_wandb_metrics(run_ids, metric='val_loss')

    else:
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")

    # Create averaged model
    averaged_model, metadata = create_weight_averaged_model(
        run_ids=run_ids,
        weights=weights,
        model_name=f"smart_averaged_{weighting_strategy}"
    )

    # Add weighting strategy to metadata
    metadata['weighting_strategy'] = weighting_strategy
    metadata['individual_weights'] = dict(zip(run_ids, weights)) if weights else None

    return averaged_model, metadata


def evaluate_model_cer(model_function, validation_loader, tokenizer, max_batches: int = 5) -> float:
    """
    Quick CER evaluation for a model.
    """
    from lib.utils.metrics import calculate_cer

    total_cer = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_loader):
            if batch_idx >= max_batches:
                break

            padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths = batch

            # Get predictions
            predictions = model_function(padded_feats, feat_lengths)

            # Get targets
            targets = []
            for i, length in enumerate(transcript_lengths):
                target_tokens = padded_golden[i][:length].tolist()
                target_text = tokenizer.decode(target_tokens)
                targets.append(target_text)

            # Calculate CER
            for pred, target in zip(predictions, targets):
                cer = calculate_cer(pred, target)
                total_cer += cer
                num_samples += 1

    return total_cer / num_samples if num_samples > 0 else float('inf')


def get_weights_from_wandb_metrics(run_ids: List[str], metric: str = 'val_loss') -> List[float]:
    """
    Extract weights from wandb metrics (e.g., validation loss).
    """
    api = wandb.Api()
    metric_values = []

    for run_id in run_ids:
        try:
            run = api.run(f"idlf25/P2/{run_id}")

            # Get the best metric value
            if metric in run.summary:
                value = run.summary[metric]
                metric_values.append(value)
            else:
                print(f"⚠️  Metric {metric} not found for {run_id}, using default weight")
                metric_values.append(1.0)

        except Exception as e:
            print(f"❌ Could not get metrics for {run_id}: {e}")
            metric_values.append(1.0)

    # Convert to weights (inverse for loss metrics)
    if 'loss' in metric.lower() or 'error' in metric.lower():
        weights = [1.0 / (val + 0.01) for val in metric_values]  # Inverse
    else:
        weights = metric_values  # Direct for accuracy-like metrics

    # Normalize
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    return weights


def compare_ensemble_methods(
    run_ids: List[str],
    validation_loader,
    tokenizer: H4Tokenizer
) -> Dict:
    """
    Compare different ensemble methods: output averaging vs weight averaging.
    """
    print("🔬 Comparing ensemble methods...")

    results = {}

    # 1. Output averaging (your original ensemble)
    print("\n📊 Testing output averaging ensemble...")
    from ensemble_asr import ensemble_from_wandb

    output_ensemble = ensemble_from_wandb(run_ids)
    output_cer = evaluate_model_cer(output_ensemble, validation_loader, tokenizer)
    results['output_averaging'] = {'cer': output_cer}
    print(f"   Output averaging CER: {output_cer:.4f}")

    # 2. Equal weight averaging
    print("\n⚖️  Testing equal weight averaging...")
    equal_avg_model, _ = create_weight_averaged_model(run_ids, save_averaged_model=False)

    # Convert to function for evaluation
    def equal_avg_function(audio_features, feature_lengths):
        device = next(equal_avg_model.parameters()).device
        audio_features = audio_features.to(device)
        feature_lengths = feature_lengths.to(device)

        encoder_output, pad_mask_src, _, _ = equal_avg_model.encode(audio_features, feature_lengths)

        from lib.decoding.sequence_generator import SequenceGenerator
        generator = SequenceGenerator(equal_avg_model, tokenizer)
        return generator.beam_search(encoder_output, pad_mask_src, beam_width=10)

    equal_cer = evaluate_model_cer(equal_avg_function, validation_loader, tokenizer)
    results['equal_weight_averaging'] = {'cer': equal_cer}
    print(f"   Equal weight averaging CER: {equal_cer:.4f}")

    # 3. Performance-based weight averaging
    print("\n🏆 Testing performance-based weight averaging...")
    perf_avg_model, perf_metadata = smart_weight_averaging(
        run_ids, validation_loader, 'performance', tokenizer
    )

    def perf_avg_function(audio_features, feature_lengths):
        device = next(perf_avg_model.parameters()).device
        audio_features = audio_features.to(device)
        feature_lengths = feature_lengths.to(device)

        encoder_output, pad_mask_src, _, _ = perf_avg_model.encode(audio_features, feature_lengths)

        from lib.decoding.sequence_generator import SequenceGenerator
        generator = SequenceGenerator(perf_avg_model, tokenizer)
        return generator.beam_search(encoder_output, pad_mask_src, beam_width=10)

    perf_cer = evaluate_model_cer(perf_avg_function, validation_loader, tokenizer)
    results['performance_weight_averaging'] = {
        'cer': perf_cer,
        'weights': perf_metadata.get('individual_weights', {})
    }
    print(f"   Performance weight averaging CER: {perf_cer:.4f}")

    # Find best method
    best_method = min(results.keys(), key=lambda k: results[k]['cer'])
    best_cer = results[best_method]['cer']

    print(f"\n🎯 Best ensemble method: {best_method}")
    print(f"   Best CER: {best_cer:.4f}")

    results['best_method'] = best_method
    results['best_cer'] = best_cer

    return results


# Example usage
if __name__ == "__main__":
    # Example run IDs
    run_ids = ["run_id_1", "run_id_2", "run_id_3"]

    # Create weight-averaged ensemble
    averaged_model, metadata = create_weight_averaged_model(
        run_ids=run_ids,
        model_name="my_weight_averaged_ensemble"
    )

    print("🎉 Weight-averaged ensemble created!")
    print(f"Model saved with {metadata['total_parameters']:,} parameters")