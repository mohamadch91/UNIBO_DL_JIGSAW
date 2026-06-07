# UNIBO DL Jigsaw

This project solves a neural jigsaw reconstruction task: given nine scrambled RGB patches of shape `28x28x3`, reconstruct the original `96x96x3` image. The implementation is contained in [patches_to_image_spec.ipynb](https://github.com/mohamadch91/UNIBO_DL_JIGSAW/blob/main/patches_to_image_spec.ipynb) and is designed to run in Google Colab with Keras/TensorFlow.

## Repository

GitHub repository: [mohamadch91/UNIBO_DL_JIGSAW](https://github.com/mohamadch91/UNIBO_DL_JIGSAW)

The training process can be inspected through the repository commit history. The summarized fine-tuning experiments and final model-selection notes are documented in [Result.md](https://github.com/mohamadch91/UNIBO_DL_JIGSAW/blob/main/Result.md).

## Task

The input image is split into a `3x3` grid. Each logical grid cell is `32x32`, but the visible patch is center-cropped to `28x28`, removing border information and making the puzzle ambiguous. The model must infer:

- which patch belongs in each grid position;
- how to reconstruct the missing borders;
- how to generate a coherent final `96x96` image.

The required metric is Mean Absolute Error (MAE), reported on the held-out test split with standard deviation.

## Data Pipeline

The notebook uses the unlabeled STL-10 binary dataset, which contains `100,000` color images at `96x96` resolution. The dataset loader caches the archive in Google Drive to avoid repeated downloads in Colab.

`PatchGenerator` creates training samples on the fly:

- load a full STL-10 image and normalize it to `[0, 1]`;
- split it into a `3x3` grid of `32x32` cells;
- center-crop each cell to `28x28`;
- randomly permute the nine patches;
- return the scrambled patch stack as input and the original image as target.

The default split is `80%` train, `10%` validation, and `10%` test.

## Architecture

The model is an end-to-end neural network with no pretrained components and fewer than 6 million trainable parameters.

### 1. Patch Encoder

A shared convolutional encoder is applied to each patch with `TimeDistributed`. It converts each `28x28x3` patch into an embedding token of size `192`.

### 2. Slot Encoding

`PuzzleSlotEncoding` adds a learned embedding for the scrambled input slot. This gives the model a stable notion of where each input patch appeared in the shuffled sequence.

### 3. Context Mixer

Three transformer-style mixer blocks let the nine patch tokens exchange global information. Each block uses:

- layer normalization;
- multi-head self-attention;
- residual connection;
- MLP projection with GELU and dropout;
- second residual connection.

This stage helps the model compare patch content and infer likely relative placement.

### 4. Balanced Grid Assignment

The placement head predicts a `9x9` score matrix mapping input patch slots to destination grid cells. `BalancedGridAssignment` applies Sinkhorn-style log-space row/column normalization so the matrix behaves like a soft permutation: each patch should map to one cell and each cell should receive one patch.

Auxiliary entropy and balance losses encourage confident, balanced assignments while preserving differentiability.

### 5. Differentiable Canvas

`DifferentiablePatchCanvas` places every patch onto the `96x96` canvas using the soft assignment matrix. It also creates an alpha channel that marks known patch pixels. This keeps placement differentiable, so the assignment and reconstruction stages can train together.

### 6. Repair Decoder

A compact convolutional encoder-decoder repairs the eroded borders and uncertain regions. It uses residual convolution blocks, downsampling to `48x48` and `24x24`, then upsampling with skip connections back to `96x96`.

The output is a repaired RGB image. A final blending layer preserves known patch pixels strongly and uses the repair decoder mainly for gaps and uncertain areas.

## Training

Training uses:

- `AdamW` optimizer;
- MAE loss;
- MAE metric;
- validation checkpointing;
- learning-rate reduction on plateau;
- early stopping with best-weight restoration;
- optional final one-epoch fit on combined train and validation data.

Weights can be loaded either from a local Colab/Drive path or from Google Drive with `gdown`, depending on the configuration flags in the notebook.

## Evaluation

The notebook reports:

- mean-patch baseline MAE;
- final test MAE;
- standard deviation of per-batch test MAE;
- total and trainable parameter count;
- qualitative visualizations of scrambled patches, target images, reconstructions, and absolute error maps.

## Files

- [patches_to_image_spec.ipynb](https://github.com/mohamadch91/UNIBO_DL_JIGSAW/blob/main/patches_to_image_spec.ipynb): main notebook with data loading, model definition, training, evaluation, and visualization.
- [Result.md](https://github.com/mohamadch91/UNIBO_DL_JIGSAW/blob/main/Result.md): fine-tuning results and model-selection notes.
- [dl_layers.png](https://github.com/mohamadch91/UNIBO_DL_JIGSAW/blob/main/dl_layers.png): layer-level architecture diagram used in the notebook.
- [dl_model.png](https://github.com/mohamadch91/UNIBO_DL_JIGSAW/blob/main/dl_model.png): end-to-end model diagram used in the notebook.
