# Comprehensive Validation Strategy for ASR Ensembles
import torch
import wandb
import yaml
import os
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json

@dataclass
class ValidationResults:
    """Container for validation results."""
    cer: float
    wer: float
    bleu: float
    predictions: List[str]
    targets: List[str]
    individual_cers: List[float]
    model_name: str
    beam_params: Dict

class EnsembleValidator:
    """Comprehensive validation suite for ASR ensembles."""

    def __init__(self, tokenizer, device='cuda'):
        self.tokenizer = tokenizer
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.validation_history = []

    def validate_single_model(
        self,
        model,
        val_loader,
        beam_params: Dict = None,
        model_name: str = "model",
        max_batches: Optional[int] = None
    ) -> ValidationResults:
        """
        Comprehensive validation of a single model.

        Args:
            model: Model or ensemble function to validate
            val_loader: Validation dataloader
            beam_params: Beam search parameters
            model_name: Name for logging
            max_batches: Maximum batches to validate (None = all)

        Returns:
            ValidationResults object with comprehensive metrics
        """
        from lib.decoding.sequence_generator import SequenceGenerator
        from lib.utils.metrics import calculate_cer, calculate_wer, calculate_bleu

        if beam_params is None:
            beam_params = {
                'beam_width': 10,
                'temperature': 1.0,
                'length_penalty': 1.0,
                'repeat_penalty': 1.1
            }

        print(f"🔍 Validating {model_name}...")
        print(f"   Beam params: {beam_params}")

        all_predictions = []
        all_targets = []
        individual_cers = []

        # Check if it's a function (ensemble) or model
        is_function = callable(model) and not hasattr(model, 'parameters')

        if not is_function:
            model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if max_batches and batch_idx >= max_batches:
                    break

                padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths = batch

                # Move to device
                padded_feats = padded_feats.to(self.device)
                feat_lengths = feat_lengths.to(self.device)

                if is_function:
                    # Ensemble function
                    batch_predictions = model(padded_feats, feat_lengths)
                else:
                    # Single model
                    encoder_output, pad_mask_src, _, _ = model.encode(padded_feats, feat_lengths)

                    # Generate with beam search
                    generator = SequenceGenerator(model, self.tokenizer)
                    batch_predictions = generator.beam_search(
                        encoder_output=encoder_output,
                        pad_mask_src=pad_mask_src,
                        **beam_params,
                        max_len=256
                    )

                # Get ground truth targets
                batch_targets = []
                for i, length in enumerate(transcript_lengths):
                    target_tokens = padded_golden[i][:length].tolist()
                    target_text = self.tokenizer.decode(target_tokens)
                    batch_targets.append(target_text)

                # Calculate individual CERs
                for pred, target in zip(batch_predictions, batch_targets):
                    cer = calculate_cer(pred, target)
                    individual_cers.append(cer)

                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)

                if (batch_idx + 1) % 10 == 0:
                    print(f"   Processed {batch_idx + 1} batches...")

        # Calculate comprehensive metrics
        total_cer = sum(individual_cers) / len(individual_cers)

        # Calculate WER and BLEU
        total_wer = sum(calculate_wer(pred, target) for pred, target in zip(all_predictions, all_targets)) / len(all_predictions)
        total_bleu = sum(calculate_bleu(pred, target) for pred, target in zip(all_predictions, all_targets)) / len(all_predictions)

        results = ValidationResults(
            cer=total_cer,
            wer=total_wer,
            bleu=total_bleu,
            predictions=all_predictions,
            targets=all_targets,
            individual_cers=individual_cers,
            model_name=model_name,
            beam_params=beam_params
        )

        print(f"✅ {model_name} Results:")
        print(f"   CER:  {total_cer:.4f}")
        print(f"   WER:  {total_wer:.4f}")
        print(f"   BLEU: {total_bleu:.4f}")

        return results

    def validate_ensemble_comparison(
        self,
        run_ids: List[str],
        val_loader,
        beam_configs: List[Dict] = None,
        save_results: bool = True
    ) -> Dict[str, ValidationResults]:
        """
        Compare different ensemble methods and beam search configurations.

        Args:
            run_ids: List of wandb run IDs
            val_loader: Validation dataloader
            beam_configs: List of beam search configurations to test
            save_results: Whether to save detailed results

        Returns:
            Dictionary of validation results for each configuration
        """
        from ensemble_asr import ensemble_from_wandb
        from weight_averaging_ensemble import create_weight_averaged_model, smart_weight_averaging

        if beam_configs is None:
            beam_configs = [
                {'beam_width': 5, 'temperature': 1.0, 'length_penalty': 1.0, 'repeat_penalty': 1.1},
                {'beam_width': 10, 'temperature': 1.0, 'length_penalty': 0.8, 'repeat_penalty': 1.1},
                {'beam_width': 15, 'temperature': 0.9, 'length_penalty': 1.0, 'repeat_penalty': 1.2},
                {'beam_width': 20, 'temperature': 0.8, 'length_penalty': 1.2, 'repeat_penalty': 1.15}
            ]

        all_results = {}

        print(f"🎯 Comprehensive Ensemble Validation")
        print(f"Models: {len(run_ids)}")
        print(f"Beam configs: {len(beam_configs)}")

        # 1. Validate individual models
        print(f"\n📊 Validating individual models...")
        for i, run_id in enumerate(run_ids):
            print(f"\n--- Model {i+1}/{len(run_ids)}: {run_id} ---")

            try:
                single_model = ensemble_from_wandb([run_id])

                for j, beam_config in enumerate(beam_configs):
                    config_name = f"single_{i+1}_beam_{j+1}"

                    results = self.validate_single_model(
                        model=single_model,
                        val_loader=val_loader,
                        beam_params=beam_config,
                        model_name=f"{run_id}_beam_{j+1}",
                        max_batches=20  # Limit for speed
                    )

                    all_results[config_name] = results

            except Exception as e:
                print(f"❌ Failed to validate {run_id}: {e}")
                continue

        # 2. Validate output averaging ensemble
        print(f"\n🤝 Validating output averaging ensemble...")
        try:
            output_ensemble = ensemble_from_wandb(run_ids)

            for j, beam_config in enumerate(beam_configs):
                config_name = f"output_ensemble_beam_{j+1}"

                results = self.validate_single_model(
                    model=output_ensemble,
                    val_loader=val_loader,
                    beam_params=beam_config,
                    model_name=f"output_ensemble_beam_{j+1}",
                    max_batches=30
                )

                all_results[config_name] = results

        except Exception as e:
            print(f"❌ Failed to validate output ensemble: {e}")

        # 3. Validate weight averaging ensemble
        print(f"\n⚖️  Validating weight averaging ensemble...")
        try:
            weight_avg_model, _ = create_weight_averaged_model(
                run_ids=run_ids,
                save_averaged_model=False
            )

            # Create function wrapper
            def weight_avg_function(audio_features, feature_lengths):
                device = next(weight_avg_model.parameters()).device
                audio_features = audio_features.to(device)
                feature_lengths = feature_lengths.to(device)

                encoder_output, pad_mask_src, _, _ = weight_avg_model.encode(audio_features, feature_lengths)

                from lib.decoding.sequence_generator import SequenceGenerator
                generator = SequenceGenerator(weight_avg_model, self.tokenizer)
                return generator.beam_search(
                    encoder_output, pad_mask_src, beam_width=10, max_len=256
                )

            for j, beam_config in enumerate(beam_configs):
                config_name = f"weight_ensemble_beam_{j+1}"

                results = self.validate_single_model(
                    model=weight_avg_function,
                    val_loader=val_loader,
                    beam_params=beam_config,
                    model_name=f"weight_ensemble_beam_{j+1}",
                    max_batches=30
                )

                all_results[config_name] = results

        except Exception as e:
            print(f"❌ Failed to validate weight ensemble: {e}")

        # 4. Find best configuration
        best_config = min(all_results.keys(), key=lambda k: all_results[k].cer)
        best_cer = all_results[best_config].cer

        print(f"\n🏆 Best Configuration: {best_config}")
        print(f"   Best CER: {best_cer:.4f}")
        print(f"   Best params: {all_results[best_config].beam_params}")

        # Save results
        if save_results:
            self.save_validation_results(all_results)

        # Log to wandb if available
        if wandb.run is not None:
            self.log_results_to_wandb(all_results)

        return all_results

    def save_validation_results(self, results: Dict[str, ValidationResults]):
        """Save validation results to files."""
        os.makedirs('validation_results', exist_ok=True)

        # Save summary
        summary = {}
        detailed_results = {}

        for config_name, result in results.items():
            summary[config_name] = {
                'cer': result.cer,
                'wer': result.wer,
                'bleu': result.bleu,
                'beam_params': result.beam_params,
                'model_name': result.model_name
            }

            detailed_results[config_name] = {
                'predictions': result.predictions[:10],  # First 10 samples
                'targets': result.targets[:10],
                'individual_cers': result.individual_cers,
                **summary[config_name]
            }

        # Save summary
        with open('validation_results/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed results
        with open('validation_results/detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # Create plots
        self.create_validation_plots(results)

        print(f"💾 Validation results saved to validation_results/")

    def create_validation_plots(self, results: Dict[str, ValidationResults]):
        """Create visualization plots for validation results."""
        plt.style.use('seaborn-v0_8')

        # Extract data for plotting
        configs = list(results.keys())
        cers = [results[config].cer for config in configs]
        wers = [results[config].wer for config in configs]
        bleus = [results[config].bleu for config in configs]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # CER comparison
        axes[0, 0].bar(range(len(configs)), cers, alpha=0.7)
        axes[0, 0].set_title('Character Error Rate (CER) Comparison')
        axes[0, 0].set_ylabel('CER')
        axes[0, 0].set_xticks(range(len(configs)))
        axes[0, 0].set_xticklabels(configs, rotation=45, ha='right')

        # WER comparison
        axes[0, 1].bar(range(len(configs)), wers, alpha=0.7, color='orange')
        axes[0, 1].set_title('Word Error Rate (WER) Comparison')
        axes[0, 1].set_ylabel('WER')
        axes[0, 1].set_xticks(range(len(configs)))
        axes[0, 1].set_xticklabels(configs, rotation=45, ha='right')

        # BLEU comparison
        axes[1, 0].bar(range(len(configs)), bleus, alpha=0.7, color='green')
        axes[1, 0].set_title('BLEU Score Comparison')
        axes[1, 0].set_ylabel('BLEU')
        axes[1, 0].set_xticks(range(len(configs)))
        axes[1, 0].set_xticklabels(configs, rotation=45, ha='right')

        # CER distribution for best model
        best_config = min(configs, key=lambda k: results[k].cer)
        best_cers = results[best_config].individual_cers
        axes[1, 1].hist(best_cers, bins=30, alpha=0.7, color='red')
        axes[1, 1].set_title(f'CER Distribution - Best Model ({best_config})')
        axes[1, 1].set_xlabel('CER')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('validation_results/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Plots saved to validation_results/metrics_comparison.png")

    def log_results_to_wandb(self, results: Dict[str, ValidationResults]):
        """Log validation results to wandb."""
        for config_name, result in results.items():
            wandb.log({
                f'validation/{config_name}/cer': result.cer,
                f'validation/{config_name}/wer': result.wer,
                f'validation/{config_name}/bleu': result.bleu
            })

        # Log best configuration
        best_config = min(results.keys(), key=lambda k: results[k].cer)
        wandb.log({
            'validation/best_config': best_config,
            'validation/best_cer': results[best_config].cer
        })

    def quick_validation(
        self,
        run_ids: List[str],
        val_loader,
        target_cer: float = 0.06
    ) -> bool:
        """
        Quick validation to check if ensemble meets target CER.

        Args:
            run_ids: List of wandb run IDs
            val_loader: Validation dataloader
            target_cer: Target CER to achieve

        Returns:
            True if target CER is achieved
        """
        from ensemble_asr import ensemble_from_wandb

        print(f"⚡ Quick validation - Target CER: {target_cer:.3f}")

        try:
            ensemble = ensemble_from_wandb(run_ids)

            results = self.validate_single_model(
                model=ensemble,
                val_loader=val_loader,
                beam_params={'beam_width': 15, 'temperature': 0.9, 'length_penalty': 1.0, 'repeat_penalty': 1.2},
                model_name="quick_ensemble",
                max_batches=10
            )

            achieved = results.cer <= target_cer

            print(f"🎯 Target {'✅ ACHIEVED' if achieved else '❌ NOT ACHIEVED'}")
            print(f"   Current CER: {results.cer:.4f}")
            print(f"   Target CER:  {target_cer:.4f}")

            return achieved

        except Exception as e:
            print(f"❌ Quick validation failed: {e}")
            return False


# Example usage functions
def validate_ensemble_pipeline(run_ids: List[str]) -> Dict:
    """Complete validation pipeline for an ensemble."""
    from lib.data.asr_dataset import ASRDataset
    from lib.data.tokenizer import H4Tokenizer
    from torch.utils.data import DataLoader

    print(f"🔍 Running complete validation pipeline...")

    # Create validation data (you'll need to adjust this based on your setup)
    config_path = "config_lightweight_fast.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    token_type = config['tokenization']['token_type']
    token_map = config['tokenization']['token_map']
    tokenizer = H4Tokenizer(token_map, token_type)

    # Create training dataset first to get global stats if needed
    train_dataset = ASRDataset(
        partition=config['data']['train_partition'],
        config=config['data'],
        tokenizer=tokenizer,
        isTrainPartition=True,
        global_stats=None
    )

    # Get global stats for validation if using global_mvn
    global_stats = None
    if config['data'].get('norm') == 'global_mvn':
        global_stats = (train_dataset.global_mean, train_dataset.global_std)

    val_dataset = ASRDataset(
        partition=config['data']['val_partition'],
        config=config['data'],
        tokenizer=tokenizer,
        isTrainPartition=False,
        global_stats=global_stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,  # Set to 0 for validation
        collate_fn=val_dataset.collate_fn
    )

    # Create validator
    validator = EnsembleValidator(tokenizer)

    # Run comprehensive validation
    results = validator.validate_ensemble_comparison(
        run_ids=run_ids,
        val_loader=val_loader,
        save_results=True
    )

    return results


if __name__ == "__main__":
    # Example usage
    run_ids = ["run_id_1", "run_id_2", "run_id_3"]

    # Run validation pipeline
    results = validate_ensemble_pipeline(run_ids)

    print("🎉 Validation pipeline completed!")
    print(f"Best configuration found with CER: {min(r.cer for r in results.values()):.4f}")