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
**Val MAE: 0.0759**
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


## Adamw with  decreased weight decay and without edge-aware loss

### Configuration

```python
cfg = {
    "epochs": 10,
    "weight_decay": 5e-4,  # Add weight decay for regularization
    "use_edge_aware_loss": False,  # Disable edge-aware loss
  
}
```

### Training and Evaluation Results

#### Results on 100 epochs

**Train MAE: 0.0292**
**Val MAE: 0.0292**
**Test MAE: 0.0374**

#### Results on 10 epochs

**Train MAE: 0.0755**
**Val MAE: 0.07534**
**Test MAE: --**

## Adamw with decreased weight decay

### Configuration

```python
cfg = {
    "epochs": 10,
    "weight_decay": 1e-4,  # Add weight decay for regularization
    "use_edge_aware_loss": False,  # Disable edge-aware loss
  
}
```

### Training and Evaluation Results


#### Results on 10 epochs

**Train MAE: 0.08**
**Val MAE: 0.07983**
**Test MAE: 0.0724**

## Adamw with new weight decay again

### Configuration

```python
cfg = {
    "epochs": 10,
    "weight_decay": 2e-4,  # Add weight decay for regularization
    "use_edge_aware_loss": False,  # Disable edge-aware loss
  
}
```

### Training and Evaluation Results


#### Results on 10 epochs

**Train MAE: 0.0786**
**Val MAE: 0.0788**
**Test MAE: ---**

## Change Sinkhorn temperature

I changed it its worse than the previous configuration, so I will keep it as it is.

## Add two more residual blocks to the repair decoder

#### Results on 10 epochs

**Train MAE: 0.0770**
**Val MAE: 0.0750**
**Test MAE: ---**

## Change in DifferentiablePatchCanvas

#### Results on 10 epochs

**Train MAE: 0.079**
**Val MAE: 0.076**
**Test MAE: ---**


## Add two more residual block

#### Results on 10 epochs

**Train MAE: 0.078**
**Val MAE: 0.079**
**Test MAE: ---**

## Conclusion

The final configuration I will continue with is the original model trained with AdamW, `weight_decay=5e-4`, and no additional changes. This means I will not use edge-aware loss, I will not change the Sinkhorn temperature, I will not change `DifferentiablePatchCanvas`, and I will not add extra residual blocks.

This choice is based on the experiments: `weight_decay=5e-4` gave the strongest long-run result, with Train MAE `0.0292`, Val MAE `0.0292`, and Test MAE `0.0374` after 100 epochs. It also keeps the model simple, which is important because most extra modifications did not give a stable improvement.

## Explanation of Experiments

- **Base model:** The base model reached Test MAE around `0.044` after 50 epochs. This shows that the original architecture is already able to learn the puzzle reconstruction task well. The 10-epoch result was much worse because the model had not trained long enough yet.

- **Weight decay with Adam:** Adding weight decay to the normal Adam optimizer gave Test MAE around `0.074` after 10 epochs. This did not clearly improve the model. The reason is that Adam does not decouple weight decay from the gradient update, so the regularization effect can be less controlled.

- **Weight decay with AdamW:** AdamW with weight decay gave similar short-run Test MAE around `0.074`, but it is a better optimizer choice for this setup because AdamW applies weight decay separately from the adaptive gradient update. This gives cleaner regularization and is more reliable for tuning.

- **AdamW with edge-aware loss:** Adding edge-aware loss made the result worse, with Test MAE around `0.076`. This probably happened because the extra edge objective pushed the model to focus too much on local patch-border details, while the main task needs globally correct patch placement. Because of this, MAE alone is a better loss for this experiment.

- **Higher weight decay `0.04` with edge-aware loss:** Increasing weight decay while using edge-aware loss did not solve the problem. The Test MAE was still worse than the best configuration. This suggests that the issue was not only overfitting; the edge-aware loss was likely adding an objective that did not match the final evaluation metric.

- **Higher weight decay `0.04` without edge-aware loss:** Removing edge-aware loss improved the short-run Test MAE to around `0.0736`, but this weight decay is still quite strong. A high value can restrict the model too much and may prevent it from learning fine details after longer training.

- **Lower weight decay `5e-4` without edge-aware loss:** This was the best direction. With 100 epochs it reached Test MAE `0.0374`, which is better than the 50-epoch base model. This likely happened because `5e-4` gives enough regularization to reduce overfitting, but it is not so large that it blocks the model from learning useful patch-position features.

- **Lower weight decay `1e-4` and `2e-4`:** These smaller values did not look better in the 10-epoch runs. They probably regularized the model too weakly compared with `5e-4`, so they did not give the same balance between learning capacity and generalization.

- **Changing Sinkhorn temperature:** Changing the Sinkhorn temperature made the result worse, so the original temperature should be kept. The likely reason is that the existing value already gives a good balance between soft assignments during training and stable patch matching.

- **Adding residual blocks:** Adding extra residual blocks did not give a stable improvement. One run with two extra blocks had a reasonable Val MAE, but the second run was worse. This suggests that the deeper repair decoder adds complexity without consistently improving generalization.

- **Changing `DifferentiablePatchCanvas`:** This change did not improve the result. The original canvas implementation is likely already sufficient, and changing it may alter the reconstruction behavior without helping the optimization objective.

Overall, the best next step is to keep the architecture and loss simple, use AdamW with `weight_decay=5e-4`, and train longer with the original Sinkhorn and canvas settings.
