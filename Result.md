# Result of Training and Evaluation

## Base Model

### Configuration

```python
cfg = {
   
    # ---- Random Seed ---
    "random_seed": 42,  # Set a random seed for reproducibility

    # --- Training HyperParams ---
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 50,
    "lr_factor": 0.5,
    "min_lr": 1e-6,
    "early_stopping_patience": 5,
    "reduce_lr_patience": 3,
    "weight_decay": None, 

    # --- Dataset Params
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "data_size": 100000,

    # --- Architecture Params ---
    "embed_dim": 192,
    "num_heads": 4,
    "key_dim": 48,
    "mlp_dim": 384,
    "dropout": 0.1,

    # --- Loss Params ---
    "use_edge_aware_loss": False,
    "edge_loss_weight": 0.10,

    # --- Sinkhorn Params ---
    "sinkhorn_temp": 0.7,
    "sinkhorn_iters": 20,
    "sinkhorn_entropy_weight": 0.003,
    "sinkhorn_balance_weight": 0.02,

    # --- Patch/Grid Params ---
    "image_size": 96,
    "patch_size": 28,
    "cell_size": 32,
    "margin": 2,



    # --- Output Params ---
    "output_dir": "/content/drive/MyDrive/jigsaw_outputs",
    "best_weights_filename": "jigsaw_best_base.weights.h5",
    "final_weights_filename": "jigsaw_final_base.weights.h5",
}
```

### Training and Evaluation Results

#### 50 epochs results

**Train MAE: 0.0.0401**
**Val MAE: 0.0436**
**Test MAE: 0.044044896960258484**

#### Results on 10 epochs

**Train MAE: 0.0763**
**Val MAE: 0.0769**
**Test MAE: --**

## Add Weight Decay with ADAM Optimizer

### Configuration

```python
cfg = {
    "epochs": 10,
    "weight_decay": 1e-4,  # Add weight decay for regularization
}
```

### Training and Evaluation Results

#### Results on 10 epochs

**Train MAE: 0.0794**
**Val MAE: --**
**Test MAE: 0.07390870898962021**

## Add Weight Decay with ADAMW Optimizer

### Configuration

```python
cfg = {
    "epochs": 10,
    "weight_decay": 0.004,  # Add weight decay for regularization
}
```

### Training and Evaluation Results

#### Results on 10 epochs

**Train MAE: 0.0752**
**Val MAE: 0.0749**
**Test MAE: 0.07400824874639511**

## Adamw with new loss configuration

### Configuration

```python
cfg = {
    "epochs": 10,
    "weight_decay": 0.004,  # Add weight decay for regularization
    "use_edge_aware_loss": True,  # Disable edge-aware loss
    "edge_loss_weight": 0.10,  # Set edge loss weight to 0.10
}
```

### Training and Evaluation Results

#### Results on 10 epochs
**Train MAE: 0.0752**
**Val MAE: 0.0749**
**Test MAE: 0.07400824874639511**
