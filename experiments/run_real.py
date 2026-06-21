"""
Experiment runner for real-world datasets (AirQuality, Traffic, etc.)
Based on Cini et al. (2023) Section 8.3.1 for AQI experiments.
"""

import time
import json
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from torch.optim.lr_scheduler import MultiStepLR
import tsl
from tsl.experiment import Experiment
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, PemsBay, MetrLA
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
from tsl.ops.connectivity import adj_to_edge_index

import lib
from lib.predictors.latent_graph_predictor import (
    LatentGraphPredictor,
    SFGraphPredictor,
    IMLEGraphPredictor,
    AIMLEGraphPredictor
)
from lib.nn.graph_module import GraphModule, IMLEGraphModule, AIMLEGraphModule
from lib.nn.tts_model import TTSModel
from lib.nn.gru_model import GRUModel
from lib.nn.tts_gat_model import TTSGATModel
from lib.nn.satorras_forecaster import SatorrasForecaster


def get_model(cfg, n_channels, n_nodes):
    """Select model class and kwargs based on config."""
    model_name = cfg.model_class

    common_kwargs = dict(
        input_size=n_channels,
        hidden_size=cfg.model.hidden_size,
        output_size=n_channels,
        horizon=cfg.horizon,
        n_nodes=n_nodes,
    )

    if model_name == 'tts_model':
        model_kwargs = {**common_kwargs,
            'enc_layers': cfg.model.enc_layers,
            'gcn_layers': cfg.model.gcn_layers,
            'dropout': cfg.model.dropout,
        }
        return TTSModel, model_kwargs
    elif model_name == 'gru_model':
        model_kwargs = {**common_kwargs,
            'enc_layers': cfg.model.enc_layers,
            'dropout': cfg.model.dropout,
        }
        return GRUModel, model_kwargs
    elif model_name == 'tts_gat_model':
        model_kwargs = {**common_kwargs,
            'enc_layers': cfg.model.enc_layers,
            'gcn_layers': cfg.model.gcn_layers,
            'heads': cfg.model.heads,
            'dropout': cfg.model.dropout,
        }
        return TTSGATModel, model_kwargs
    elif model_name == 'satorras_forecaster':
        model_kwargs = {**common_kwargs,
            'window': cfg.window,
            'encoder_type': getattr(cfg.model, 'encoder_type', 'mlp'),
            'enc_layers': cfg.model.enc_layers,
            'gnn_layers': cfg.model.gcn_layers,
            'dropout': cfg.model.dropout,
            'activation': getattr(cfg.model, 'activation', 'silu'),
            'use_node_id': getattr(cfg.model, 'use_node_id', True),
            'node_emb_size': getattr(cfg.model, 'node_emb_size', None),
            'use_gate': getattr(cfg.model, 'use_gate', False),
        }
        return SatorrasForecaster, model_kwargs
    else:
        raise ValueError(f"Unknown model_class: {model_name}")


def get_dataset(cfg):
    """Load real-world dataset based on config."""
    dataset_name = cfg.dataset.name

    if dataset_name == 'air_quality':
        # Load Beijing AirQuality dataset (36 nodes with small=True)
        small = getattr(cfg.dataset, 'small', True)
        dataset = AirQuality(root=lib.config['data_dir'], small=small)
        return dataset

    elif dataset_name == 'pems_bay':
        # Load PEMS-BAY traffic dataset (325 nodes)
        #mask_zeros = getattr(cfg.dataset, 'mask_zeros', True)
        mask_zeros = getattr(cfg.dataset, 'impute_zeros', True)
        dataset = PemsBay(root=lib.config['data_dir'], mask_zeros=mask_zeros)
        return dataset

    elif dataset_name == 'metr_la':
        # Load METR-LA traffic dataset (207 nodes)
        impute_zeros = getattr(cfg.dataset, 'impute_zeros', True)
        dataset = MetrLA(root=lib.config['data_dir'], impute_zeros=impute_zeros)
        return dataset

    else:
        raise ValueError(f"Dataset {dataset_name} not available.")


def get_connectivity(dataset, cfg):
    """
    Build connectivity for the dataset.

    Options:
    - 'distance': Use distance-based connectivity (thresholded)
    - 'knn': K-nearest neighbors based on distance
    - 'none': No prior connectivity (learn from scratch)
    """
    conn_type = getattr(cfg.dataset, 'connectivity_type', 'none')

    if conn_type == 'none':
        return None

    elif conn_type == 'distance':
        # Get distance matrix and threshold
        threshold = getattr(cfg.dataset, 'distance_threshold', 0.1)
        dist = dataset.get_similarity('distance')
        # Convert to torch tensor if numpy
        if isinstance(dist, np.ndarray):
            dist = torch.from_numpy(dist).float()
        # Normalize distances to [0, 1]
        dist_norm = dist / dist.max()
        # Create adjacency: 1 if distance < threshold
        adj = (dist_norm < threshold).float()
        # Remove self-loops
        adj.fill_diagonal_(0)
        edge_index, edge_weight = adj_to_edge_index(adj)
        return (edge_index, edge_weight)

    elif conn_type == 'knn':
        k = getattr(cfg.dataset, 'knn_k', 5)
        dist = dataset.get_similarity('distance')
        # Convert to torch tensor if numpy
        if isinstance(dist, np.ndarray):
            dist = torch.from_numpy(dist).float()
        n_nodes = dist.shape[0]
        adj = torch.zeros(n_nodes, n_nodes)
        # For each node, connect to k nearest neighbors
        for i in range(n_nodes):
            distances = dist[i].clone()
            distances[i] = float('inf')  # Exclude self
            _, indices = torch.topk(distances, k, largest=False)
            adj[i, indices] = 1.0
        # Make symmetric
        adj = ((adj + adj.T) > 0).float()
        edge_index, edge_weight = adj_to_edge_index(adj)
        return (edge_index, edge_weight)

    else:
        raise ValueError(f"Connectivity type {conn_type} not supported.")


def run_experiment(cfg):
    """Run the latent graph learning experiment on real data."""

    original_run_dir = cfg.run.dir
    num_runs = cfg.num_runs
    start_run = getattr(cfg, 'start_run', 1)  # 1-indexed, default = start from run_1
    complexity_metrics = {i: {} for i in range(start_run, num_runs + 1)}

    for run_idx in range(start_run - 1, num_runs):
        print(f"\n{'='*60}")
        print(f"Starting run {run_idx+1}/{num_runs}")
        print(f"{'='*60}")
        cfg.run.dir = f"{original_run_dir}/run_{run_idx+1}"

        # Load dataset
        dataset = get_dataset(cfg)

        # Get number of nodes
        n_nodes = dataset.n_nodes

        # Select graph module class based on mode
        if cfg.graph_mode == 'imle':
            gm_class = IMLEGraphModule
        elif cfg.graph_mode == 'aimle':
            gm_class = AIMLEGraphModule
        else:
            gm_class = GraphModule

        ########################################
        # Data module setup                    #
        ########################################

        # Get optional connectivity for the dataset
        connectivity = get_connectivity(dataset, cfg)

        # Create mask for missing values
        mask = dataset.mask

        # Create SpatioTemporalDataset
        torch_dataset = SpatioTemporalDataset(
            target=dataset.dataframe(),
            connectivity=connectivity,
            mask=mask,
            horizon=cfg.horizon,
            window=cfg.window,
            stride=cfg.stride
        )

        # Create data module with standard scaling
        splitter = dataset.get_splitter(
            val_len=cfg.dataset.splits.val_len,
            test_len=cfg.dataset.splits.test_len
        )

        dm = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers={'target': StandardScaler(axis=(0, 1))},
            splitter=splitter,
            batch_size=cfg.batch_size,
            workers=cfg.workers
        )

        dm.setup()

        ########################################
        # Model and Predictor setup            #
        ########################################

        # Graph module kwargs
        gm_kwargs = dict(
            n_nodes=n_nodes,
            mode=cfg.graph_mode
        )
        gm_kwargs.update(cfg.graph_module.hparams)

        # Loss function and metrics
        loss_fn = MaskedMAE()

        # Base metrics (averaged across all horizons)
        metrics = {
            'mae': MaskedMAE(),
            'mse': MaskedMSE(),
            'mape': MaskedMAPE()
        }

        # Add horizon-specific metrics for traffic datasets (GTS paper evaluation)
        # Both PEMS-BAY and METR-LA use 5-min intervals: 15min=step 3, 30min=step 6, 60min=step 12
        if cfg.dataset.name in ['pems_bay', 'metr_la']:
            metrics.update({
                'mae_at_15': MaskedMAE(at=2),   # 3rd step (15 min)
                'mae_at_30': MaskedMAE(at=5),   # 6th step (30 min)
                'mae_at_60': MaskedMAE(at=11),  # 12th step (60 min)
                'mape_at_15': MaskedMAPE(at=2),
                'mape_at_30': MaskedMAPE(at=5),
                'mape_at_60': MaskedMAPE(at=11),
                'mse_at_15': MaskedMSE(at=2),   # RMSE = sqrt(mse_at_15) post-hoc
                'mse_at_30': MaskedMSE(at=5),   # RMSE = sqrt(mse_at_30) post-hoc
                'mse_at_60': MaskedMSE(at=11),  # RMSE = sqrt(mse_at_60) post-hoc
            })

        # Set eval mode based on graph mode
        if cfg.graph_mode == 'sf':
            if cfg.use_baseline == 'frechet':
                eval_mode = 'mode'
            else:
                eval_mode = 'sampling'
        else:
            eval_mode = 'sampling'

        # Configure predictor based on graph mode
        if cfg.graph_mode == 'pd' or cfg.graph_mode == 'st':
            predictor_class = LatentGraphPredictor
            pred_kwargs = dict(
                graph_module_class=gm_class,
                graph_module_kwargs=gm_kwargs,
                mc_samples=cfg.mc_samples,
            )

        elif cfg.graph_mode == 'sf':
            predictor_class = SFGraphPredictor
            pred_kwargs = dict(
                sf_weight=cfg.sf_weight,
                graph_module_class=gm_class,
                graph_module_kwargs=gm_kwargs,
                use_baseline=cfg.use_baseline,
                mc_samples=cfg.mc_samples,
                eval_mode=eval_mode,
                variance_reduced=cfg.variance_reduced,
                surrogate_lam=cfg.lam,
            )
            if cfg.use_baseline == 'doublecv':
                doublecv_args = {
                    'gradient_level': cfg.gradient_level,
                    'doublecv_mode': cfg.doublecv_mode,
                    'use_taylor': cfg.use_taylor
                }
                pred_kwargs.update(doublecv_args)

        elif cfg.graph_mode == 'imle':
            predictor_class = IMLEGraphPredictor
            pred_kwargs = dict(
                graph_module_class=gm_class,
                graph_module_kwargs=gm_kwargs,
                mc_samples=cfg.mc_samples,
                eval_mode=eval_mode,
                clip_grad=cfg.clip_grad,
                clip_grad_val=0.5
            )

        elif cfg.graph_mode == 'aimle':
            predictor_class = AIMLEGraphPredictor
            pred_kwargs = dict(
                graph_module_class=gm_class,
                graph_module_kwargs=gm_kwargs,
                mc_samples=cfg.mc_samples,
                eval_mode=eval_mode,
                clip_grad=cfg.clip_grad,
                clip_grad_val=0.5
            )

        else:
            raise NotImplementedError(f"Graph learning mode {cfg.graph_mode} not available.")

        # Model configuration (selected via config)
        model_cls, model_kwargs = get_model(cfg, torch_dataset.n_channels, n_nodes)

        # Create predictor
        predictor = predictor_class(
            model_class=model_cls,
            model_kwargs=model_kwargs,
            optim_class=torch.optim.Adam,
            optim_kwargs=dict(cfg.optimizer.hparams),
            loss_fn=loss_fn,
            metrics=metrics,
            scheduler_class=MultiStepLR if getattr(cfg, 'scheduler', None) else None,
            scheduler_kwargs=dict(cfg.scheduler) if getattr(cfg, 'scheduler', None) else None,
            scale_target=False,
            **pred_kwargs
        )

        print(f'Eval mode for test: {predictor.eval_mode}')

        ########################################
        # Training                             #
        ########################################

        early_stop_callback = EarlyStopping(
            monitor='val_mae',
            patience=cfg.patience,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.run.dir,
            save_top_k=1,
            monitor='val_mae',
            mode='min',
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        batches_epoch = 1.0 if cfg.batches_epoch < 0 else cfg.batches_epoch
        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            limit_train_batches=batches_epoch,
            default_root_dir=cfg.run.dir,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
            gradient_clip_algorithm='norm',  # Clip global gradient norm (default in PL)
            gradient_clip_val=getattr(cfg, 'clip_grad_val', 5.0) if cfg.clip_grad else None,
        )

        print("Checking model parameters for requires_grad:")
        for name, param in predictor.named_parameters():
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

        # Time and memory tracking
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Train
        trainer.fit(
            predictor,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader()
        )

        actual_epochs = trainer.current_epoch + 1  # Adding 1 since epochs are 0-indexed
        training_end_time = time.time()
        training_total_time = training_end_time - start_time

        ########################################
        # Testing                              #
        ########################################

        trainer.test(
            ckpt_path=checkpoint_callback.best_model_path,
            dataloaders=dm.test_dataloader()
        )

        # Record metrics
        final_time = time.time()
        total_time = final_time - start_time
        peak_memory_usage = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

        complexity_metrics[run_idx+1]['total_time'] = total_time
        complexity_metrics[run_idx+1]['total_memory'] = peak_memory_usage
        complexity_metrics[run_idx+1]['training_time'] = training_total_time
        complexity_metrics[run_idx+1]['actual_epochs'] = actual_epochs
        complexity_metrics[run_idx+1]['time_per_epoch'] = complexity_metrics[run_idx+1]['training_time'] / complexity_metrics[run_idx+1]['actual_epochs']

    # Save summary
    if start_run > 1:
        summary_file = f"{original_run_dir}/complexity_summary_updated.json"
    else:
        summary_file = f"{original_run_dir}/complexity_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(complexity_metrics, f, indent=2, default=str)


if __name__ == '__main__':
    exp = Experiment(run_fn=run_experiment, config_path='config/real')
    exp.run()
