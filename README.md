# Jigsaw Pipeline Code Explanation

This document explains the deep learning pipeline that solves the Jigsaw reconstruction task end-to-end. 

**The Challenge**: 
The original `PatchGenerator` randomly scrambles the patches and throws away the permutations, only returning `X` (9 scrambled patches) and `Y` (the original reconstructed image).
To train the model without modifying the generator or adding a new one, we cannot simply build a "permutation classifier" because we don't have the labels (the true positions) to calculate the loss!

**The Solution**: 
We use a **Differentiable Assembly Layer**. Instead of using `argmax` to select which patch goes to which slot, we use `softmax` probabilities to calculate a weighted sum of the patches. This means gradients can flow continuously from the final image reconstruction loss (MAE) all the way back through the assembly process and into the permutation predictor.

---

## 1. Differentiable Assembly Layer

This custom layer reorders the patches using soft attention, meaning the reordering operation is fully differentiable.

```python
@tf.keras.utils.register_keras_serializable()
class DifferentiableAssemblyLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # We define 9 learnable vectors (embeddings), one for each of the 9 target slots in the 3x3 grid.
        self.slot_embeddings = self.add_weight(
            shape=(9, 256),
            initializer="random_normal",
            trainable=True,
            name="slot_embeddings"
        )
        
    def call(self, inputs):
        # We receive the 9 raw patches and their 256-dimensional contextual features.
        patches, patch_features = inputs 
        B = tf.shape(patches)[0] # Extract the batch size
        
        # tf.einsum calculates the dot product between the 9 target slots and the 9 patch features.
        # This outputs a similarity matrix (logits) of shape (Batch, 9 slots, 9 patches).
        logits = tf.einsum('sc,bpc->bsp', self.slot_embeddings, patch_features)
        
        # We apply softmax over the patches. For each slot, this creates a probability distribution
        # indicating which patch is most likely to belong there.
        P = tf.nn.softmax(logits, axis=-1) 
        
        # Here we perform the soft assembly! We multiply the probabilities by the actual image patches
        # and sum them up. If the network is confident, P is close to 1 for the correct patch and 0 for others.
        soft_patches = tf.einsum('bsp,bphwi->bshwi', P, patches) 
        
        # The original grid patches were 32x32, but we were given 28x28 center crops (2px erosion).
        # We pad our 28x28 patches with 2 pixels of zeros on all sides to restore them to 32x32.
        padded_patches = tf.pad(soft_patches, [[0,0], [0,0], [2,2], [2,2], [0,0]])
        
        # We reshape the 9 patches into a 3x3 grid of 32x32 patches.
        grid_img = tf.reshape(padded_patches, (B, 3, 3, 32, 32, 3))
        # We transpose the axes to properly interleave the rows and columns into a single 96x96 image.
        grid_img = tf.transpose(grid_img, [0, 1, 3, 2, 4, 5])
        assembled_img = tf.reshape(grid_img, (B, 96, 96, 3))
        
        # Next, we build a binary mask (4th channel) to tell the U-Net where the valid pixels are (1)
        # and where the missing padded borders are (0).
        mask_patch = tf.ones((28, 28, 1))
        padded_mask = tf.pad(mask_patch, [[2,2], [2,2], [0,0]])
        padded_mask = tf.broadcast_to(padded_mask, [B, 9, 32, 32, 1])
        grid_mask = tf.reshape(padded_mask, (B, 3, 3, 32, 32, 1))
        grid_mask = tf.transpose(grid_mask, [0, 1, 3, 2, 4, 5])
        assembled_mask = tf.reshape(grid_mask, (B, 96, 96, 1))
        
        # We concatenate the 96x96 RGB image with the 96x96 mask, outputting a (96, 96, 4) tensor.
        return tf.concat([assembled_img, assembled_mask], axis=-1)
```

## 2. Jigsaw Solver Architecture

The solver pipeline combines a CNN patch encoder, a Transformer context network, the differentiable assembly layer, and a Refinement U-Net.

```python
def build_jigsaw_solver():
    # Input is the batch of 9 scrambled patches (B, 9, 28, 28, 3)
    inputs = layers.Input(shape=(9, 28, 28, 3))
    
    # === 1. Patch Encoder (CNN) ===
    # We define a standard CNN to encode a single 28x28 patch into a 1D feature vector.
    encoder_input = layers.Input(shape=(28, 28, 3))
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # GlobalAveragePooling reduces the spatial dimensions (2x2) down to a single 256-d vector.
    x = layers.GlobalAveragePooling2D()(x)
    patch_encoder = models.Model(encoder_input, x, name="patch_encoder")
    
    # TimeDistributed applies the exact same CNN weights independently to all 9 patches.
    # Output is (B, 9 patches, 256 features).
    encoded_patches = layers.TimeDistributed(patch_encoder)(inputs) 
    
    # === 2. Transformer Context ===
    x = encoded_patches
    # We use 3 layers of Multi-Head Self-Attention.
    # This allows each patch to "look" at the other 8 patches and compare its edges/features 
    # to figure out its relative position in the larger puzzle.
    for _ in range(3):
        attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
        ffn_out = layers.Dense(512, activation='relu')(x)
        ffn_out = layers.Dense(256)(ffn_out)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)
        
    patch_features = x # These are the context-aware patch features.
    
    # === 3. Differentiable Assembly ===
    # We pass the raw patches and their contextual features to our custom assembly layer.
    # It outputs a rough, unrefined 96x96 image with black grid lines.
    assembled_concat = DifferentiableAssemblyLayer()([inputs, patch_features])
    
    # === 4. U-Net Refinement ===
    # The U-Net takes the rough 96x96x4 assembled image (RGB + Mask) and refines it.
    # It downsamples the image while increasing channels...
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(assembled_concat)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1) # Down to 48x48
    
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2) # Down to 24x24
    
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3) # Down to 12x12
    
    # Bottleneck at the bottom of the U-Net
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)
    
    # ...and then upsamples it back, using Concatenate skip-connections to retain sharp details 
    # from the original patches, allowing it to accurately in-paint the missing 2px grid lines.
    u5 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c4) 
    u5 = layers.Concatenate()([u5, c3])
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(c5)
    
    u6 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c5) 
    u6 = layers.Concatenate()([u6, c2])
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c6)
    u7 = layers.Concatenate()([u7, c1])
    c7 = layers.Conv2D(32, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, 3, activation='relu', padding='same')(c7)
    
    # Final output layer uses Sigmoid to ensure pixel values stay bounded between [0, 1].
    outputs = layers.Conv2D(3, 1, activation='sigmoid', name="output_image")(c7)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="jigsaw_solver")
    return model
```

## 3. Evaluation & Visualization

Because we are training end-to-end, we can simply compile the model against the MAE of the output image. The loss naturally backpropagates to adjust the permutation probabilities. 

At the bottom of the notebook, we also include a function to randomly visualize the results:

```python
# Visualization Function
def visualize_random_predictions(model, generator, num_samples=3):
    import random
    
    # 1. Pick a random batch index from the generator
    batch_idx = random.randint(0, len(generator) - 1)
    
    # 2. Extract the scrambled patches (X_batch) and original ground-truth images (Y_batch)
    X_batch, Y_batch = generator[batch_idx]
    
    # 3. Ask the model to reconstruct the full images from the scrambled patches
    preds = model.predict(X_batch, verbose=0)
    
    # 4. Pick `num_samples` random indices within this specific batch for visualization
    actual_batch_size = len(X_batch)
    sample_indices = random.sample(range(actual_batch_size), min(num_samples, actual_batch_size))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    for i, idx in enumerate(sample_indices):
        orig_img = Y_batch[idx]
        pred_img = preds[idx]
        
        # 5. Calculate the MAE (Mean Absolute Error) for this specific image
        mae = np.mean(np.abs(orig_img - pred_img))
        
        # 6. Plot the Original Image
        axes[i][0].imshow(orig_img)
        axes[i][0].set_title("Original Image")
        axes[i][0].axis('off')
        
        # 7. Plot the Model's Reconstruction alongside its MAE
        axes[i][1].imshow(pred_img)
        axes[i][1].set_title(f"Reconstructed (MAE: {mae:.4f})")
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.show()
```
