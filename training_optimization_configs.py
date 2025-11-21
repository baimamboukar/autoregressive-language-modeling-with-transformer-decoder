# Training Optimization Configurations for Maximum Performance
import yaml
import os
from typing import Dict, List
import copy

def create_optimization_configs():
    """
    Create optimized training configurations for different scenarios.
    """

    # Base lightweight config
    base_config = {
        'Name': "optimized_base",
        'tokenization': {
            'token_type': "5k",
            'token_map': {
                'char': 'hw4lib/data/tokenizer_jsons/tokenizer_char.json',
                '1k'  : 'hw4lib/data/tokenizer_jsons/tokenizer_1000.json',
                '5k'  : 'hw4lib/data/tokenizer_jsons/tokenizer_5000.json',
                '10k' : 'hw4lib/data/tokenizer_jsons/tokenizer_10000.json'
            }
        },
        'data': {
            'root': "/content/hw4_data/hw4p2_data",
            'train_partition': "train-clean-100",
            'val_partition': "dev-clean",
            'test_partition': "test-clean",
            'subset': 1.0,
            'batch_size': 32,
            'NUM_WORKERS': 4,
            'norm': 'global_mvn',
            'num_feats': 80,
            'specaug': True,
            'specaug_conf': {
                'apply_freq_mask': True,
                'freq_mask_width_range': 12,
                'num_freq_mask': 2,
                'apply_time_mask': True,
                'time_mask_width_range': 60,
                'num_time_mask': 2
            }
        },
        'model': {
            'input_dim': 80,
            'time_reduction': 8,
            'reduction_method': 'both',
            'd_model': 256,
            'num_encoder_layers': 4,
            'num_decoder_layers': 4,
            'num_encoder_heads': 8,
            'num_decoder_heads': 8,
            'd_ff_encoder': 1024,
            'd_ff_decoder': 1024,
            'skip_encoder_pe': False,
            'skip_decoder_pe': False,
            'dropout': 0.1,
            'layer_drop_rate': 0.05,
            'weight_tying': True
        },
        'training': {
            'use_wandb': True,
            'wandb_run_id': "none",
            'resume': True,
            'gradient_accumulation_steps': 1,
            'wandb_project': "HW4P2"
        },
        'loss': {
            'label_smoothing': 0.15,
            'ctc_weight': 0.3
        },
        'optimizer': {
            'name': "adamw",
            'lr': 0.003,
            'weight_decay': 0.01,
            'adamw': {
                'betas': [0.9, 0.98],
                'eps': 1.0e-6,
                'amsgrad': False
            }
        },
        'scheduler': {
            'name': "cosine",
            'cosine': {
                'T_max': 20,
                'eta_min': 0.00001,
                'last_epoch': -1
            },
            'warmup': {
                'enabled': True,
                'type': "exponential",
                'epochs': 3,
                'start_factor': 0.1,
                'end_factor': 1.0
            }
        },
        'beam_search': {
            'configs': {
                'beam_5': {
                    'beam_width': 5,
                    'temperature': 1.0,
                    'repeat_penalty': 1.1,
                    'length_penalty': 0.8
                },
                'beam_10': {
                    'beam_width': 10,
                    'temperature': 0.9,
                    'repeat_penalty': 1.15,
                    'length_penalty': 0.9
                },
                'beam_15': {
                    'beam_width': 15,
                    'temperature': 0.85,
                    'repeat_penalty': 1.2,
                    'length_penalty': 1.0
                }
            }
        }
    }

    configs = {}

    # 1. Fast Convergence Config
    fast_config = copy.deepcopy(base_config)
    fast_config.update({
        'Name': "fast_convergence_15m",
        'optimizer': {
            'name': "adamw",
            'lr': 0.005,  # Higher LR
            'weight_decay': 0.005,  # Lower weight decay
            'adamw': {
                'betas': [0.9, 0.999],  # Different beta2
                'eps': 1.0e-6,
                'amsgrad': False
            }
        },
        'scheduler': {
            'name': "cosine",
            'cosine': {
                'T_max': 12,  # Shorter cycle
                'eta_min': 0.00005,
                'last_epoch': -1
            },
            'warmup': {
                'enabled': True,
                'type': "exponential",
                'epochs': 1,  # Minimal warmup
                'start_factor': 0.2,
                'end_factor': 1.0
            }
        },
        'loss': {
            'label_smoothing': 0.1,  # Less smoothing for faster convergence
            'ctc_weight': 0.25
        }
    })
    configs['fast_convergence'] = fast_config

    # 2. Stability-Focused Config
    stable_config = copy.deepcopy(base_config)
    stable_config.update({
        'Name': "stable_training_15m",
        'optimizer': {
            'name': "adamw",
            'lr': 0.0015,  # Lower, stable LR
            'weight_decay': 0.02,  # Higher regularization
            'adamw': {
                'betas': [0.9, 0.98],
                'eps': 1.0e-8,  # More stable
                'amsgrad': True  # More stable variant
            }
        },
        'scheduler': {
            'name': "cosine",
            'cosine': {
                'T_max': 30,  # Longer cycle
                'eta_min': 0.000005,  # Very low minimum
                'last_epoch': -1
            },
            'warmup': {
                'enabled': True,
                'type': "exponential",
                'epochs': 5,  # Longer warmup
                'start_factor': 0.05,
                'end_factor': 1.0
            }
        },
        'model': {
            **base_config['model'],
            'dropout': 0.15,  # Higher dropout
            'layer_drop_rate': 0.1  # More regularization
        },
        'loss': {
            'label_smoothing': 0.2,  # More smoothing
            'ctc_weight': 0.4
        }
    })
    configs['stable_training'] = stable_config

    # 3. Mixed Precision Optimized Config
    mixed_precision_config = copy.deepcopy(base_config)
    mixed_precision_config.update({
        'Name': "mixed_precision_15m",
        'data': {
            **base_config['data'],
            'batch_size': 48,  # Larger batch with mixed precision
        },
        'optimizer': {
            'name': "adamw",
            'lr': 0.004,  # Slightly higher for larger batch
            'weight_decay': 0.008,
            'adamw': {
                'betas': [0.9, 0.98],
                'eps': 1.0e-4,  # Adjusted for mixed precision
                'amsgrad': False
            }
        },
        'training': {
            **base_config['training'],
            'mixed_precision': True,  # Enable if supported
            'gradient_clip_norm': 1.0  # Gradient clipping
        }
    })
    configs['mixed_precision'] = mixed_precision_config

    # 4. High Performance Config (for final models)
    high_perf_config = copy.deepcopy(base_config)
    high_perf_config.update({
        'Name': "high_performance_25m",
        'model': {
            **base_config['model'],
            'd_model': 320,  # Larger model
            'num_encoder_layers': 5,
            'num_decoder_layers': 5,
            'd_ff_encoder': 1280,
            'd_ff_decoder': 1280,
            'dropout': 0.08,
            'layer_drop_rate': 0.02
        },
        'data': {
            **base_config['data'],
            'batch_size': 24,  # Smaller batch for larger model
            'gradient_accumulation_steps': 2,
            'specaug_conf': {
                'apply_freq_mask': True,
                'freq_mask_width_range': 15,
                'num_freq_mask': 3,
                'apply_time_mask': True,
                'time_mask_width_range': 70,
                'num_time_mask': 3
            }
        },
        'optimizer': {
            'name': "adamw",
            'lr': 0.002,
            'weight_decay': 0.015,
            'adamw': {
                'betas': [0.9, 0.98],
                'eps': 1.0e-6,
                'amsgrad': False
            }
        },
        'scheduler': {
            'name': "cosine",
            'cosine': {
                'T_max': 25,
                'eta_min': 0.000001,
                'last_epoch': -1
            },
            'warmup': {
                'enabled': True,
                'type': "exponential",
                'epochs': 4,
                'start_factor': 0.1,
                'end_factor': 1.0
            }
        },
        'loss': {
            'label_smoothing': 0.12,
            'ctc_weight': 0.35
        }
    })
    configs['high_performance'] = high_perf_config

    # 5. Fine-tuning Config (for second stage training)
    finetune_config = copy.deepcopy(base_config)
    finetune_config.update({
        'Name': "finetune_stage2_15m",
        'optimizer': {
            'name': "adamw",
            'lr': 0.0005,  # Much lower LR
            'weight_decay': 0.005,
            'adamw': {
                'betas': [0.9, 0.999],  # Different beta for fine-tuning
                'eps': 1.0e-8,
                'amsgrad': True
            }
        },
        'scheduler': {
            'name': "cosine",
            'cosine': {
                'T_max': 8,  # Short cycle
                'eta_min': 0.00001,
                'last_epoch': -1
            },
            'warmup': {
                'enabled': False  # No warmup for fine-tuning
            }
        },
        'model': {
            **base_config['model'],
            'dropout': 0.05,  # Lower dropout for fine-tuning
            'layer_drop_rate': 0.01
        },
        'loss': {
            'label_smoothing': 0.05,  # Minimal smoothing
            'ctc_weight': 0.2
        }
    })
    configs['finetune_stage2'] = finetune_config

    # 6. Data Augmentation Heavy Config
    aug_heavy_config = copy.deepcopy(base_config)
    aug_heavy_config.update({
        'Name': "aug_heavy_15m",
        'data': {
            **base_config['data'],
            'specaug_conf': {
                'apply_freq_mask': True,
                'freq_mask_width_range': 20,  # Heavy augmentation
                'num_freq_mask': 4,
                'apply_time_mask': True,
                'time_mask_width_range': 100,
                'num_time_mask': 4
            }
        },
        'optimizer': {
            'name': "adamw",
            'lr': 0.002,  # Lower LR to handle heavy augmentation
            'weight_decay': 0.02,
            'adamw': {
                'betas': [0.9, 0.98],
                'eps': 1.0e-6,
                'amsgrad': False
            }
        },
        'model': {
            **base_config['model'],
            'dropout': 0.2,  # Higher dropout with heavy aug
            'layer_drop_rate': 0.15
        },
        'loss': {
            'label_smoothing': 0.25,  # Higher smoothing
            'ctc_weight': 0.3
        }
    })
    configs['aug_heavy'] = aug_heavy_config

    return configs

def save_optimization_configs(configs: Dict):
    """Save all optimization configs to files."""
    os.makedirs("optimization_configs", exist_ok=True)

    for name, config in configs.items():
        filename = f"optimization_configs/config_{name}.yaml"
        with open(filename, 'w') as f:
            yaml.dump(config, f, indent=2)
        print(f"✅ Saved: {filename}")

def create_training_schedule() -> Dict:
    """
    Create a strategic training schedule for optimal results.
    """
    schedule = {
        'phase_1_parallel_base_training': {
            'description': "Train base models in parallel with different configs",
            'configs': [
                'config_variant_1_balanced.yaml',
                'config_variant_2_wider.yaml',
                'config_variant_3_deeper.yaml',
                'optimization_configs/config_fast_convergence.yaml'
            ],
            'epochs': 20,
            'parallel': True,
            'goal': "Get diverse base models quickly"
        },
        'phase_2_stability_training': {
            'description': "Train stable, longer models for robustness",
            'configs': [
                'optimization_configs/config_stable_training.yaml',
                'optimization_configs/config_high_performance.yaml'
            ],
            'epochs': 30,
            'parallel': True,
            'goal': "Get high-quality robust models"
        },
        'phase_3_fine_tuning': {
            'description': "Fine-tune best models from previous phases",
            'configs': ['optimization_configs/config_finetune_stage2.yaml'],
            'epochs': 10,
            'parallel': False,
            'goal': "Polish best models for maximum performance",
            'prerequisites': "Best run_ids from phases 1&2"
        },
        'phase_4_ensemble': {
            'description': "Create and validate ensemble",
            'methods': [
                'output_averaging',
                'weight_averaging_equal',
                'weight_averaging_performance'
            ],
            'beam_search_tuning': True,
            'goal': "Achieve CER < 6%"
        }
    }

    return schedule

def create_quick_wins_config():
    """Create a configuration focused on quick wins and fast iteration."""
    quick_wins = {
        'Name': "quick_wins_10m",
        'tokenization': {
            'token_type': "1k",  # Smaller vocab for speed
            'token_map': {
                'char': 'hw4lib/data/tokenizer_jsons/tokenizer_char.json',
                '1k'  : 'hw4lib/data/tokenizer_jsons/tokenizer_1000.json',
                '5k'  : 'hw4lib/data/tokenizer_jsons/tokenizer_5000.json',
                '10k' : 'hw4lib/data/tokenizer_jsons/tokenizer_10000.json'
            }
        },
        'data': {
            'root': "/content/hw4_data/hw4p2_data",
            'train_partition': "train-clean-100",
            'val_partition': "dev-clean",
            'test_partition': "test-clean",
            'subset': 0.7,  # Use less data for speed
            'batch_size': 48,
            'NUM_WORKERS': 4,
            'norm': 'global_mvn',
            'num_feats': 80,
            'specaug': True,
            'specaug_conf': {
                'apply_freq_mask': True,
                'freq_mask_width_range': 8,  # Light augmentation
                'num_freq_mask': 2,
                'apply_time_mask': True,
                'time_mask_width_range': 40,
                'num_time_mask': 2
            }
        },
        'model': {
            'input_dim': 80,
            'time_reduction': 8,
            'reduction_method': 'conv',  # Fastest reduction
            'd_model': 224,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'num_encoder_heads': 8,
            'num_decoder_heads': 8,
            'd_ff_encoder': 896,
            'd_ff_decoder': 896,
            'skip_encoder_pe': False,
            'skip_decoder_pe': False,
            'dropout': 0.1,
            'layer_drop_rate': 0.0,  # No layer drop for speed
            'weight_tying': True
        },
        'training': {
            'use_wandb': True,
            'wandb_run_id': "none",
            'resume': True,
            'gradient_accumulation_steps': 1,
            'wandb_project': "HW4P2"
        },
        'loss': {
            'label_smoothing': 0.1,
            'ctc_weight': 0.3
        },
        'optimizer': {
            'name': "adamw",
            'lr': 0.005,  # High LR for fast convergence
            'weight_decay': 0.005,
            'adamw': {
                'betas': [0.9, 0.999],
                'eps': 1.0e-6,
                'amsgrad': False
            }
        },
        'scheduler': {
            'name': "cosine",
            'cosine': {
                'T_max': 8,  # Short cycle
                'eta_min': 0.0001,
                'last_epoch': -1
            },
            'warmup': {
                'enabled': True,
                'type': "exponential",
                'epochs': 1,
                'start_factor': 0.2,
                'end_factor': 1.0
            }
        },
        'beam_search': {
            'configs': {
                'beam_5': {
                    'beam_width': 5,
                    'temperature': 1.0,
                    'repeat_penalty': 1.1,
                    'length_penalty': 0.8
                }
            }
        }
    }

    return quick_wins

if __name__ == "__main__":
    print("🔧 Creating training optimization configurations...")

    # Create all configs
    configs = create_optimization_configs()

    # Add quick wins config
    configs['quick_wins'] = create_quick_wins_config()

    # Save all configs
    save_optimization_configs(configs)

    # Create training schedule
    schedule = create_training_schedule()

    # Save training schedule
    with open("training_schedule.yaml", 'w') as f:
        yaml.dump(schedule, f, indent=2)

    print(f"\n🎯 Created {len(configs)} optimization configurations:")
    for name, config in configs.items():
        print(f"  ✅ {name}: {config['Name']}")

    print(f"\n📋 Training schedule saved to: training_schedule.yaml")
    print(f"📂 All configs saved to: optimization_configs/")

    print(f"\n🚀 Quick start recommendations:")
    print(f"  1. Fast iteration: config_quick_wins.yaml")
    print(f"  2. Best balance: config_fast_convergence.yaml")
    print(f"  3. Maximum quality: config_high_performance.yaml")
    print(f"  4. Most stable: config_stable_training.yaml")

    print(f"\n🎯 Strategic approach:")
    print(f"  1. Start with quick_wins for rapid prototyping")
    print(f"  2. Train 3-4 variants in parallel")
    print(f"  3. Fine-tune best models")
    print(f"  4. Create ensemble for CER < 6%")