# Beam Search Parameter Tuning for Optimal CER
import torch
import wandb
from typing import List, Dict, Tuple, Optional
from itertools import product
import numpy as np

def tune_beam_search_params(
    model_or_ensemble,
    val_loader,
    tokenizer,
    param_grid: Dict = None,
    max_combinations: int = 20,
    save_results: bool = True
) -> Dict:
    """
    Systematically tune beam search parameters to find optimal CER.

    Args:
        model_or_ensemble: Either a single model or ensemble function
        val_loader: Validation dataloader
        tokenizer: H4Tokenizer instance
        param_grid: Parameter grid for tuning
        max_combinations: Maximum parameter combinations to test
        save_results: Whether to save results to file

    Returns:
        Dictionary with best parameters and all results
    """
    from hw4lib.decoding.sequence_generator import SequenceGenerator
    from hw4lib.utils.metrics import calculate_cer

    # Default parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'beam_width': [5, 10, 15, 20, 25],
            'temperature': [0.7, 0.8, 0.9, 1.0, 1.1],
            'length_penalty': [0.6, 0.8, 1.0, 1.2, 1.4],
            'repeat_penalty': [1.0, 1.1, 1.15, 1.2, 1.25]
        }

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    # Limit combinations if too many
    if len(all_combinations) > max_combinations:
        print(f"⚠️  Too many combinations ({len(all_combinations)}), sampling {max_combinations}")
        np.random.shuffle(all_combinations)
        all_combinations = all_combinations[:max_combinations]

    print(f"🔍 Testing {len(all_combinations)} beam search parameter combinations")

    results = []
    best_cer = float('inf')
    best_params = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i, param_combo in enumerate(all_combinations):
        # Create parameter dict
        params = dict(zip(param_names, param_combo))

        print(f"\n📊 Test {i+1}/{len(all_combinations)}")
        print(f"   Params: {params}")

        try:
            # Test parameters
            cer = evaluate_with_beam_params(
                model_or_ensemble,
                val_loader,
                tokenizer,
                **params
            )

            results.append({
                'params': params.copy(),
                'cer': cer
            })

            print(f"   CER: {cer:.4f}")

            # Update best
            if cer < best_cer:
                best_cer = cer
                best_params = params.copy()
                print(f"   🎯 New best CER: {cer:.4f}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue

    # Sort results by CER
    results.sort(key=lambda x: x['cer'])

    print(f"\n🏆 Best Results:")
    print(f"   Best CER: {best_cer:.4f}")
    print(f"   Best params: {best_params}")

    print(f"\n📈 Top 5 configurations:")
    for i, result in enumerate(results[:5]):
        print(f"   {i+1}. CER: {result['cer']:.4f} | {result['params']}")

    # Save results
    if save_results:
        import json
        with open(f"beam_search_tuning_results.json", "w") as f:
            json.dump({
                'best_params': best_params,
                'best_cer': best_cer,
                'all_results': results
            }, f, indent=2)
        print(f"💾 Results saved to beam_search_tuning_results.json")

    return {
        'best_params': best_params,
        'best_cer': best_cer,
        'all_results': results
    }


def evaluate_with_beam_params(
    model_or_ensemble,
    val_loader,
    tokenizer,
    beam_width: int = 10,
    temperature: float = 1.0,
    length_penalty: float = 1.0,
    repeat_penalty: float = 1.0,
    max_batches: int = 10
) -> float:
    """
    Evaluate model with specific beam search parameters.

    Args:
        model_or_ensemble: Model or ensemble function to evaluate
        val_loader: Validation dataloader
        tokenizer: H4Tokenizer instance
        beam_width: Beam width for search
        temperature: Temperature for sampling
        length_penalty: Length penalty factor
        repeat_penalty: Repetition penalty factor
        max_batches: Maximum batches to evaluate (for speed)

    Returns:
        Character Error Rate (CER)
    """
    from hw4lib.decoding.sequence_generator import SequenceGenerator
    from hw4lib.utils.metrics import calculate_cer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_predictions = []
    all_targets = []

    # Check if it's an ensemble function or single model
    is_ensemble = callable(model_or_ensemble) and hasattr(model_or_ensemble, '__name__')

    if not is_ensemble:
        # Single model - create generator
        model_or_ensemble.eval()
        generator = SequenceGenerator(model_or_ensemble, tokenizer)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths = batch

            # Move to device
            padded_feats = padded_feats.to(device)
            feat_lengths = feat_lengths.to(device)

            if is_ensemble:
                # Ensemble function
                predictions = model_or_ensemble(
                    padded_feats,
                    feat_lengths,
                    return_logits=False
                )
                # Ensemble returns list of strings
                batch_predictions = predictions
            else:
                # Single model
                encoder_output, pad_mask_src, _, _ = model_or_ensemble.encode(padded_feats, feat_lengths)

                # Generate with beam search
                batch_predictions = generator.beam_search(
                    encoder_output=encoder_output,
                    pad_mask_src=pad_mask_src,
                    beam_width=beam_width,
                    max_len=256,
                    temperature=temperature,
                    length_penalty=length_penalty,
                    repeat_penalty=repeat_penalty
                )

            # Get ground truth targets
            batch_targets = []
            for i, length in enumerate(transcript_lengths):
                target_tokens = padded_golden[i][:length].tolist()
                target_text = tokenizer.decode(target_tokens)
                batch_targets.append(target_text)

            all_predictions.extend(batch_predictions)
            all_targets.extend(batch_targets)

    # Calculate CER
    total_cer = 0.0
    for pred, target in zip(all_predictions, all_targets):
        cer = calculate_cer(pred, target)
        total_cer += cer

    avg_cer = total_cer / len(all_predictions)
    return avg_cer


def quick_beam_search_sweep(
    run_ids: List[str],
    fast_mode: bool = True
) -> Dict:
    """
    Quick beam search parameter sweep for multiple models.

    Args:
        run_ids: List of wandb run IDs to test
        fast_mode: If True, use smaller parameter grid for speed

    Returns:
        Dictionary with results for each model
    """
    from ensemble_asr import ensemble_from_wandb

    if fast_mode:
        param_grid = {
            'beam_width': [5, 10, 15],
            'temperature': [0.8, 1.0, 1.2],
            'length_penalty': [0.8, 1.0, 1.2],
            'repeat_penalty': [1.0, 1.1, 1.2]
        }
        max_combinations = 10
    else:
        param_grid = {
            'beam_width': [5, 10, 15, 20, 25],
            'temperature': [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            'length_penalty': [0.6, 0.8, 1.0, 1.2, 1.4],
            'repeat_penalty': [1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
        }
        max_combinations = 30

    results_per_model = {}

    # Test individual models
    for run_id in run_ids:
        print(f"\n🔍 Tuning beam search for model: {run_id}")

        try:
            # Create single-model ensemble
            single_model = ensemble_from_wandb([run_id])

            # Tune parameters
            results = tune_beam_search_params(
                model_or_ensemble=single_model,
                val_loader=None,  # Will need to create this
                tokenizer=None,   # Will need to create this
                param_grid=param_grid,
                max_combinations=max_combinations,
                save_results=False
            )

            results_per_model[run_id] = results

        except Exception as e:
            print(f"❌ Failed to tune {run_id}: {e}")
            continue

    # Test ensemble
    if len(run_ids) > 1:
        print(f"\n🤝 Tuning beam search for ensemble of {len(run_ids)} models")

        try:
            ensemble = ensemble_from_wandb(run_ids)

            ensemble_results = tune_beam_search_params(
                model_or_ensemble=ensemble,
                val_loader=None,  # Will need to create this
                tokenizer=None,   # Will need to create this
                param_grid=param_grid,
                max_combinations=max_combinations,
                save_results=True  # Save ensemble results
            )

            results_per_model['ensemble'] = ensemble_results

        except Exception as e:
            print(f"❌ Failed to tune ensemble: {e}")

    return results_per_model


def get_optimal_beam_configs() -> Dict[str, Dict]:
    """
    Return pre-tuned optimal beam search configurations for different scenarios.
    """
    return {
        'conservative': {
            'beam_width': 10,
            'temperature': 1.0,
            'length_penalty': 0.8,
            'repeat_penalty': 1.1,
            'description': 'Safe, balanced configuration'
        },
        'aggressive': {
            'beam_width': 20,
            'temperature': 0.8,
            'length_penalty': 1.2,
            'repeat_penalty': 1.2,
            'description': 'More exploration, longer sequences'
        },
        'fast': {
            'beam_width': 5,
            'temperature': 1.0,
            'length_penalty': 1.0,
            'repeat_penalty': 1.0,
            'description': 'Fast inference, good baseline'
        },
        'diverse': {
            'beam_width': 15,
            'temperature': 1.2,
            'length_penalty': 1.0,
            'repeat_penalty': 1.05,
            'description': 'Higher temperature for diversity'
        }
    }


# Example usage
if __name__ == "__main__":
    # Example: Tune beam search for your models
    run_ids = ["your_run_id_1", "your_run_id_2", "your_run_id_3"]

    # Quick sweep
    results = quick_beam_search_sweep(run_ids, fast_mode=True)

    print("🎯 Beam search tuning completed!")
    for model_id, result in results.items():
        if result:
            print(f"{model_id}: Best CER = {result['best_cer']:.4f}")
            print(f"  Best params: {result['best_params']}")