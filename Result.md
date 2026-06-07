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
**Train MAE: 0.0765**
**Val MAE: 0.0805**
**Test MAE: 0.0764131173491478**

## Adamw with  increased weight decay and loss configuration

### Configuration

```python
cfg = {
    "epochs": 10,
    "weight_decay": 0.04,  # Add weight decay for regularization
    "use_edge_aware_loss": True,  # Disable edge-aware loss
    "edge_loss_weight": 0.10,  # Set edge loss weight to 0.10
}
```

### Training and Evaluation Results

#### Results on 10 epochs
**Train MAE: 0.0781**
**Val MAE: 0.0757**
**Test MAE: 0.07548248022794724**

## Adamw with  increased weight decay and without edge-aware loss

### Configuration

```python
cfg = {
    "epochs": 10,
    "weight_decay": 0.04,  # Add weight decay for regularization
    "use_edge_aware_loss": False,  # Disable edge-aware loss
  
}
```

### Training and Evaluation Results

#### Results on 10 epochs
**Train MAE: 0.0760**
**Val MAE: 0.0760**
**Test MAE: 0.07358548790216446**


## Conclusion

- The base model with 50 epochs achieved a test MAE of approximately 0.044, which is a good starting point.
- Adding weight decay with the ADAM optimizer did not improve the performance, resulting in a test MAE of approximately 0.074.
- Switching to the ADAMW optimizer with weight decay did better than all the previous configurations, achieving a test MAE of approximately 0.074.
- However, adding the edge-aware loss did not improve the performance and resulted in a test MAE of approximately 0.076, which is worse than the previous configuration without the edge-aware loss. This suggests that the edge-aware loss may not be beneficial for this specific task or may require further
- I decided to remove the edge-aware loss and only use the MAE loss, as it seems to be more effective for this task based on the results obtained.
  Also going forward with the ADAMW optimizer with weight decay, as it showed better performance compared to the ADAM optimizer.
- We need to tune weight decay but it seems that the best value is around 0.004, as it provided the best results among the tested values.
- After fine tuning the weight decay, I think best value is around 0.04, as it provided the best results among the tested values.
  