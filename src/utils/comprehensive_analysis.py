"""
=============================================================================
COMPREHENSIVE TECHNICAL ANALYSIS - ViT Chest X-ray Classification Project
=============================================================================
Senior AI Engineer Technical Review Report
Author: Technical Review System
Date: Auto-generated

This script performs a complete end-to-end analysis of the ViT Chest X-ray
classification project, evaluating architecture, data, training, and inference.
"""

import os
import sys
import glob
import random
import math
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from collections import Counter

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: PROJECT OVERVIEW & CONFIGURATION
# ============================================================================

print("="*80)
print("COMPREHENSIVE TECHNICAL ANALYSIS - ViT Chest X-ray Classification")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Configure paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
CSV_PATH = os.path.join(INPUT_DIR, "Data_Entry_2017_v2020.csv")

# ============================================================================
# SECTION 2: DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: DATA ANALYSIS")
print("="*80)

# Load metadata
print("\n[2.1] Loading Dataset Metadata...")
df = pd.read_csv(CSV_PATH)
print(f"   Total records in CSV: {len(df):,}")
print(f"   CSV columns: {list(df.columns)}")

# Count available images
image_files = glob.glob(os.path.join(INPUT_DIR, "*.png"))
print(f"\n[2.2] Available Images: {len(image_files):,}")

# Filter to available images only
available_image_names = set(os.path.basename(f) for f in image_files)
df_available = df[df['Image Index'].isin(available_image_names)]
print(f"   Matched records: {len(df_available):,}")

# Disease label analysis
print("\n[2.3] Disease Distribution Analysis:")
all_labels = []
for labels in df_available['Finding Labels']:
    all_labels.extend(labels.split('|'))

label_counts = Counter(all_labels)
print("\n   Disease Class Distribution:")
print("   " + "-"*50)
total_labels = sum(label_counts.values())
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    pct = count / total_labels * 100
    bar = "█" * int(pct / 2)
    print(f"   {label:<25} {count:>6} ({pct:5.1f}%) {bar}")

# Check class imbalance
print("\n[2.4] Class Imbalance Analysis:")
max_count = max(label_counts.values())
min_count = min(label_counts.values())
imbalance_ratio = max_count / min_count
print(f"   Maximum class count: {max_count:,} ({max(label_counts, key=label_counts.get)})")
print(f"   Minimum class count: {min_count:,} ({min(label_counts, key=label_counts.get)})")
print(f"   Imbalance ratio: {imbalance_ratio:.1f}x")
if imbalance_ratio > 10:
    print("   ⚠️  SEVERE class imbalance detected - requires attention!")

# Multi-label analysis
print("\n[2.5] Multi-Label Analysis:")
label_counts_per_image = df_available['Finding Labels'].apply(lambda x: len(x.split('|')))
print(f"   Images with single label: {sum(label_counts_per_image == 1):,} ({sum(label_counts_per_image == 1)/len(df_available)*100:.1f}%)")
print(f"   Images with multi-labels: {sum(label_counts_per_image > 1):,} ({sum(label_counts_per_image > 1)/len(df_available)*100:.1f}%)")
print(f"   Max labels per image: {max(label_counts_per_image)}")
print(f"   Avg labels per image: {label_counts_per_image.mean():.2f}")

# Image properties analysis
print("\n[2.6] Image Properties Analysis:")
sample_images = random.sample(image_files, min(10, len(image_files)))
sizes = []
for img_path in sample_images:
    img = cv2.imread(img_path)
    if img is not None:
        sizes.append(img.shape)
        
if sizes:
    print(f"   Sample image shape: {sizes[0]}")
    print(f"   Pixel value range: 0-255 (8-bit)")
    print(f"   Channels: 3 (RGB/BGR)")

# ============================================================================
# SECTION 3: MODEL ARCHITECTURE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: MODEL ARCHITECTURE ANALYSIS")
print("="*80)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention, Add, Flatten, Input
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
    print(f"\n[3.1] TensorFlow Version: {tf.__version__}")
except ImportError:
    TF_AVAILABLE = False
    print("\n[3.1] TensorFlow not available")

try:
    import torch
    import timm
    TORCH_AVAILABLE = True
    print(f"[3.1] PyTorch Version: {torch.__version__}")
    print(f"[3.1] TIMM Version: {timm.__version__}")
except ImportError:
    TORCH_AVAILABLE = False

if TF_AVAILABLE:
    print("\n[3.2] Vision Transformer Architecture (from scratch):")
    
    # Define architecture parameters
    input_shape = (224, 224, 3)
    patch_size = 32
    num_patches = (224 // patch_size) ** 2  # 49 patches
    projection_dim = 64
    num_heads = 4
    transformer_layers = 8
    mlp_head_units = [2048, 1024]
    num_classes = 15  # 14 diseases + No Finding
    
    print(f"""
   Architecture Configuration:
   ┌─────────────────────────────────────────────────────────────┐
   │ Input Shape:         {str(input_shape):<36} │
   │ Patch Size:          {patch_size}×{patch_size} pixels                            │
   │ Number of Patches:   {num_patches} (7×7 grid)                          │
   │ Projection Dim:      {projection_dim}                                     │
   │ Transformer Layers:  {transformer_layers}                                      │
   │ Attention Heads:     {num_heads}                                      │
   │ MLP Units:           {mlp_head_units}                             │
   │ Output Classes:      {num_classes} (multi-label sigmoid)               │
   └─────────────────────────────────────────────────────────────┘
    """)
    
    # Create Patches layer
    class Patches(layers.Layer):
        def __init__(self, patch_size):
            super().__init__()
            self.patch_size = patch_size
            
        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID',
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches
    
    # Create PatchEncoder layer
    class PatchEncoder(layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super().__init__()
            self.num_patches = num_patches
            self.projection_dim = projection_dim
            self.projection = Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )
            
        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            positions = tf.expand_dims(positions, axis=0)
            projected_patches = self.projection(patch)
            encoded = projected_patches + self.position_embedding(positions)
            return encoded
    
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = Dense(units, activation=tf.nn.gelu)(x)
            x = Dropout(dropout_rate)(x)
        return x
    
    def create_vit_classifier():
        inputs = Input(shape=input_shape)
        patches = Patches(patch_size)(inputs)
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
        
        transformer_units = [projection_dim * 2, projection_dim]
        
        for _ in range(transformer_layers):
            x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            x2 = Add()([attention_output, encoded_patches])
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            encoded_patches = Add()([x3, x2])
        
        representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = Flatten()(representation)
        representation = Dropout(0.5)(representation)
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        logits = Dense(num_classes, activation='sigmoid')(features)
        
        model = Model(inputs=inputs, outputs=logits)
        return model
    
    print("[3.3] Building ViT Model...")
    vit_model = create_vit_classifier()
    
    print("\n[3.4] Model Summary:")
    total_params = vit_model.count_params()
    trainable_params = sum([tf.reduce_prod(v.shape).numpy() for v in vit_model.trainable_variables])
    
    print(f"""
   ┌─────────────────────────────────────────────────────────────┐
   │ Total Parameters:      {total_params:>15,}                  │
   │ Trainable Parameters:  {trainable_params:>15,}                  │
   │ Model Size (approx):   {total_params * 4 / 1024 / 1024:.1f} MB (float32)                    │
   └─────────────────────────────────────────────────────────────┘
    """)
    
    # Layer-by-layer analysis
    print("[3.5] Layer Architecture Analysis:")
    print("   " + "-"*60)
    for i, layer in enumerate(vit_model.layers[:15]):  # Show first 15 layers
        layer_params = sum([tf.reduce_prod(w.shape).numpy() for w in layer.weights])
        print(f"   {i:2d}. {layer.name:<30} | Params: {layer_params:>10,}")
    print("   ...")
    print(f"   Total layers: {len(vit_model.layers)}")
    
    # Test forward pass
    print("\n[3.6] Forward Pass Test:")
    test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    start_time = time.time()
    test_output = vit_model.predict(test_input, verbose=0)
    inference_time = (time.time() - start_time) * 1000
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {test_output.shape}")
    print(f"   Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
    print(f"   Inference time: {inference_time:.2f} ms")
    print("   ✓ Forward pass successful!")

# ============================================================================
# SECTION 4: TRAINING CONFIGURATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: TRAINING CONFIGURATION ANALYSIS")
print("="*80)

print("""
[4.1] Training Hyperparameters:
   ┌─────────────────────────────────────────────────────────────┐
   │ Batch Size:          32                                     │
   │ Learning Rate:       1e-4                                   │
   │ Weight Decay:        1e-6                                   │
   │ Epochs:              10                                     │
   │ Optimizer:           AdamW                                  │
   │ Loss Function:       Binary Cross-Entropy                   │
   │ Metrics:             Accuracy, AUC                          │
   └─────────────────────────────────────────────────────────────┘

[4.2] Data Augmentation Pipeline:
   ┌─────────────────────────────────────────────────────────────┐
   │ Training Augmentation:                                      │
   │   • Resize to 224×224                                       │
   │   • Random Horizontal Flip (p=0.5)                          │
   │   • Random Rotation (±5°)                                   │
   │   • Color Jitter (brightness=0.1, contrast=0.1)             │
   │   • Normalize (ImageNet mean/std)                           │
   │                                                             │
   │ Validation/Test Transform:                                  │
   │   • Resize to 224×224                                       │
   │   • Normalize (ImageNet mean/std)                           │
   └─────────────────────────────────────────────────────────────┘

[4.3] Callbacks:
   • ModelCheckpoint (save best model on val_loss)
   • ReduceLROnPlateau (ViT-v2 only)
   • EarlyStopping (ViT-v2 only, patience=5)

[4.4] Data Split:
   • Training:   60%
   • Validation: 20%
   • Test:       20%
""")

# ============================================================================
# SECTION 5: EVALUATION METRICS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: EVALUATION METRICS ANALYSIS")
print("="*80)

print("""
[5.1] Metrics Used:
   ┌─────────────────────────────────────────────────────────────┐
   │ • Binary Accuracy (per-label accuracy)                      │
   │ • AUC (Area Under ROC Curve)                                │
   │ • Binary Cross-Entropy Loss                                 │
   │ • ROC Curves (per-class visualization)                      │
   └─────────────────────────────────────────────────────────────┘

[5.2] Metrics Analysis:
   
   For multi-label classification:
   - Binary Accuracy treats each label independently
   - AUC measures ranking quality across threshold choices
   - ROC curves visualize true positive vs false positive rates

[5.3] Metric Appropriateness:
   ✓ Binary Cross-Entropy: Appropriate for multi-label sigmoid output
   ✓ AUC: Good for imbalanced datasets
   ⚠️  Missing: F1-score, Precision, Recall, Specificity
   ⚠️  Missing: Class-weighted metrics for imbalanced data
""")

# ============================================================================
# SECTION 6: COMPARATIVE MODEL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: COMPARATIVE MODEL ANALYSIS")
print("="*80)

print("""
[6.1] Models in Project:

   ┌────────────────────────────────────────────────────────────────────────────┐
   │ Model        │ Framework  │ Approach           │ Parameters  │ Pretrained │
   ├────────────────────────────────────────────────────────────────────────────┤
   │ CNN          │ TensorFlow │ Simple Conv layers │ ~1M         │ No         │
   │ ResNet       │ TensorFlow │ Residual blocks    │ ~2-5M       │ No         │
   │ ViT-v1       │ TensorFlow │ Transformer scratch│ ~9.4M       │ No         │
   │ ViT-v2       │ TensorFlow │ ViT + Regularization│ ~9.4M      │ No         │
   │ ViT-ResNet   │ PyTorch    │ Pre-trained ViT    │ ~86M        │ Yes        │
   └────────────────────────────────────────────────────────────────────────────┘

[6.2] Architecture Comparison:

   CNN (Baseline):
   • 2 Conv2D layers (32, 64 filters)
   • MaxPooling after each conv
   • Dense layer (512 units)
   • Simple, fast, but limited capacity

   ResNet (Custom):
   • Residual blocks with skip connections
   • GlobalAveragePooling
   • Better gradient flow than CNN

   ViT-v1 (From Scratch):
   • 8 Transformer encoder layers
   • 4 attention heads
   • 49 patches (32×32)
   • Learned positional embeddings

   ViT-v2 (Enhanced):
   • Same as ViT-v1
   • + L2 Regularization (0.01)
   • + Learning rate scheduling
   • + Early stopping

   ViT-ResNet (Pre-trained):
   • vit_base_patch16_224 from timm
   • 86M parameters (ImageNet pretrained)
   • Fine-tuning approach
""")

# ============================================================================
# SECTION 7: STRENGTHS AND WEAKNESSES
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: STRENGTHS AND WEAKNESSES ANALYSIS")
print("="*80)

print("""
[7.1] STRENGTHS (Pros):

   Architecture:
   ✓ Multiple model implementations for comparison
   ✓ Proper ViT implementation with patches and position encoding
   ✓ Pre-trained model option (ViT-ResNet) for transfer learning
   ✓ Multi-label classification with sigmoid activation

   Data Processing:
   ✓ Proper data augmentation pipeline
   ✓ Train/Val/Test split implemented
   ✓ Both TensorFlow and PyTorch implementations
   ✓ DatasetParser class for flexible data handling

   Training:
   ✓ AdamW optimizer with weight decay
   ✓ Model checkpointing for best model
   ✓ ROC curve visualization
   ✓ Multi-metric evaluation (Accuracy, AUC)

[7.2] WEAKNESSES (Cons):

   Critical Issues:
   ✗ SEVERE class imbalance not addressed (No Finding: 60%+)
   ✗ No class weighting in loss function
   ✗ Missing imports in notebooks (keras not imported in cnn.ipynb)
   ✗ Deprecated API usage (ops.expand_dims should be tf.expand_dims)
   ✗ Small sample size in data.ipynb (100 samples) for demo

   Architecture Issues:
   ✗ ViT from scratch may underperform without massive data
   ✗ Patch size 32×32 may lose fine-grained details
   ✗ No attention visualization for interpretability
   ✗ Missing gradient checkpointing for memory efficiency

   Data Issues:
   ✗ No stratified sampling for multi-label data
   ✗ Limited augmentation for medical images
   ✗ No test-time augmentation (TTA)
   ✗ Mixed frameworks (TF/PyTorch) adds complexity

   Evaluation Issues:
   ✗ Missing F1, Precision, Recall metrics
   ✗ No confusion matrix analysis
   ✗ No per-class performance breakdown
   ✗ No statistical significance testing
""")

# ============================================================================
# SECTION 8: QUALITY ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("SECTION 8: QUALITY ASSESSMENT")
print("="*80)

print("""
[8.1] Correctness Assessment:
   
   ┌─────────────────────────────────────────────────────────────┐
   │ Aspect                  │ Score │ Notes                    │
   ├─────────────────────────────────────────────────────────────┤
   │ Model Architecture      │  8/10 │ Correct ViT implementation│
   │ Data Pipeline          │  7/10 │ Works but has issues     │
   │ Training Loop          │  7/10 │ Standard implementation   │
   │ Evaluation Metrics     │  6/10 │ Missing important metrics │
   │ Code Quality           │  5/10 │ Missing imports, deprecated│
   └─────────────────────────────────────────────────────────────┘

[8.2] Performance Assessment:

   ┌─────────────────────────────────────────────────────────────┐
   │ Aspect                  │ Score │ Notes                    │
   ├─────────────────────────────────────────────────────────────┤
   │ Model Capacity         │  8/10 │ 9.4M params is adequate   │
   │ Training Efficiency    │  6/10 │ No mixed precision        │
   │ Inference Speed        │  7/10 │ ~50-100ms per image       │
   │ Memory Usage           │  6/10 │ No gradient checkpointing │
   └─────────────────────────────────────────────────────────────┘

[8.3] Robustness Assessment:

   ┌─────────────────────────────────────────────────────────────┐
   │ Aspect                  │ Score │ Notes                    │
   ├─────────────────────────────────────────────────────────────┤
   │ Class Imbalance        │  3/10 │ NOT HANDLED              │
   │ Data Augmentation      │  6/10 │ Basic augmentation only   │
   │ Regularization         │  7/10 │ Dropout + L2 (v2)        │
   │ Early Stopping         │  5/10 │ Only in ViT-v2           │
   └─────────────────────────────────────────────────────────────┘

[8.4] Maintainability Assessment:

   ┌─────────────────────────────────────────────────────────────┐
   │ Aspect                  │ Score │ Notes                    │
   ├─────────────────────────────────────────────────────────────┤
   │ Code Organization      │  6/10 │ Separate notebooks good   │
   │ Documentation          │  5/10 │ Minimal comments          │
   │ Error Handling         │  4/10 │ Few try/except blocks    │
   │ Configuration          │  5/10 │ Hardcoded values         │
   └─────────────────────────────────────────────────────────────┘

[8.5] Scalability Assessment:

   ┌─────────────────────────────────────────────────────────────┐
   │ Aspect                  │ Score │ Notes                    │
   ├─────────────────────────────────────────────────────────────┤
   │ Data Scalability       │  6/10 │ DataLoader implemented    │
   │ Model Scalability      │  7/10 │ ViT scales with data     │
   │ Multi-GPU Support      │  3/10 │ Not implemented          │
   │ Distributed Training   │  2/10 │ Not implemented          │
   └─────────────────────────────────────────────────────────────┘

[8.6] Reproducibility Assessment:

   ┌─────────────────────────────────────────────────────────────┐
   │ Aspect                  │ Score │ Notes                    │
   ├─────────────────────────────────────────────────────────────┤
   │ Random Seed Setting    │  5/10 │ Partial (only split)     │
   │ Requirements File      │  7/10 │ Present but incomplete   │
   │ Environment Setup      │  4/10 │ Version conflicts        │
   │ Data Versioning        │  6/10 │ CSV with metadata        │
   └─────────────────────────────────────────────────────────────┘

   OVERALL QUALITY SCORE: 5.8/10
""")

# ============================================================================
# SECTION 9: RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("SECTION 9: IMPROVEMENT RECOMMENDATIONS")
print("="*80)

print("""
[9.1] SHORT-TERM FIXES (Immediate):

   Priority 1 - Bug Fixes:
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. Add missing imports in cnn.ipynb and resnet.ipynb       │
   │    - from tensorflow import keras                           │
   │    - from tensorflow.keras.models import Sequential         │
   │    - from tensorflow.keras.layers import Conv2D, Dense, etc │
   │                                                             │
   │ 2. Fix deprecated API in ViT notebooks:                     │
   │    - Replace ops.expand_dims → tf.expand_dims               │
   │    - Replace ops.arange → tf.range                          │
   │                                                             │
   │ 3. Add random seed for reproducibility:                     │
   │    - np.random.seed(42)                                     │
   │    - tf.random.set_seed(42)                                 │
   │    - torch.manual_seed(42)                                  │
   └─────────────────────────────────────────────────────────────┘

   Priority 2 - Data Handling:
   ┌─────────────────────────────────────────────────────────────┐
   │ 4. Increase sample size from 100 to full dataset or larger │
   │    - df = parser.sample(10000, is_weighted=True)           │
   │                                                             │
   │ 5. Add class weights for imbalanced data:                   │
   │    - pos_weight = (n_negative / n_positive) per class      │
   │    - Use weighted_cross_entropy or focal_loss               │
   └─────────────────────────────────────────────────────────────┘

[9.2] MEDIUM-TERM IMPROVEMENTS (1-2 weeks):

   Architecture Improvements:
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. Add attention map visualization for interpretability    │
   │                                                             │
   │ 2. Implement hybrid CNN-Transformer architecture:          │
   │    - Use CNN backbone for feature extraction               │
   │    - Apply transformer on CNN features                      │
   │                                                             │
   │ 3. Add Grad-CAM visualization for explainability           │
   │                                                             │
   │ 4. Reduce patch size to 16×16 for finer details            │
   └─────────────────────────────────────────────────────────────┘

   Training Improvements:
   ┌─────────────────────────────────────────────────────────────┐
   │ 5. Implement focal loss for class imbalance:               │
   │    FL(p) = -α(1-p)^γ * log(p)                              │
   │                                                             │
   │ 6. Add more comprehensive metrics:                          │
   │    - F1-score (macro, micro, weighted)                     │
   │    - Precision/Recall per class                            │
   │    - Confusion matrix                                       │
   │                                                             │
   │ 7. Implement learning rate warmup:                          │
   │    - Linear warmup for first N epochs                       │
   │    - Cosine annealing schedule                              │
   │                                                             │
   │ 8. Add mixed precision training (FP16):                     │
   │    - tf.keras.mixed_precision.set_global_policy('mixed_float16')│
   └─────────────────────────────────────────────────────────────┘

[9.3] LONG-TERM ENHANCEMENTS (1-2 months):

   Advanced Features:
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. Implement ensemble of models:                            │
   │    - CNN + ViT ensemble                                     │
   │    - Multiple ViT with different patch sizes                │
   │                                                             │
   │ 2. Add test-time augmentation (TTA):                        │
   │    - Average predictions over augmented versions            │
   │                                                             │
   │ 3. Implement k-fold cross-validation:                       │
   │    - More robust performance estimation                     │
   │                                                             │
   │ 4. Add contrastive learning pretraining:                    │
   │    - Self-supervised pretraining on unlabeled data          │
   │                                                             │
   │ 5. Implement multi-GPU/distributed training:                │
   │    - tf.distribute.MirroredStrategy                         │
   │    - PyTorch DistributedDataParallel                        │
   └─────────────────────────────────────────────────────────────┘

   Production Readiness:
   ┌─────────────────────────────────────────────────────────────┐
   │ 6. Add model export for deployment:                         │
   │    - SavedModel format (TensorFlow)                         │
   │    - ONNX export for cross-platform                         │
   │    - TensorRT optimization for inference                    │
   │                                                             │
   │ 7. Create inference API:                                    │
   │    - FastAPI or Flask endpoint                              │
   │    - Batch inference support                                │
   │    - Input validation                                       │
   │                                                             │
   │ 8. Add monitoring and logging:                              │
   │    - MLflow or Weights & Biases integration                 │
   │    - TensorBoard logging                                    │
   │    - Model versioning                                       │
   └─────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# SECTION 10: VISUAL ANALYSIS (Save figures)
# ============================================================================

print("\n" + "="*80)
print("SECTION 10: GENERATING VISUALIZATIONS")
print("="*80)

output_dir = os.path.join(PROJECT_ROOT, "analysis_output")
os.makedirs(output_dir, exist_ok=True)

# 1. Disease Distribution Plot
print("\n[10.1] Creating Disease Distribution Chart...")
fig, ax = plt.subplots(figsize=(14, 8))
labels_sorted = sorted(label_counts.items(), key=lambda x: -x[1])
diseases = [x[0] for x in labels_sorted]
counts = [x[1] for x in labels_sorted]

colors = plt.cm.viridis(np.linspace(0, 0.9, len(diseases)))
bars = ax.barh(diseases, counts, color=colors)
ax.set_xlabel('Count', fontsize=12)
ax.set_title('Disease Distribution in Available Dataset', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add count labels
for bar, count in zip(bars, counts):
    ax.text(count + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
            f'{count:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_disease_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: {os.path.join(output_dir, '01_disease_distribution.png')}")

# 2. ViT Architecture Diagram
print("[10.2] Creating ViT Architecture Diagram...")
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Draw pipeline
boxes = [
    (5, 50, 15, 'Input\n224×224×3', '#E8F4FD'),
    (22, 50, 15, 'Patches\n49×3072', '#D4EDDA'),
    (39, 50, 15, 'Linear\nProjection\n49×64', '#FFF3CD'),
    (56, 50, 15, 'Transformer\nEncoder\n×8 layers', '#F8D7DA'),
    (73, 50, 15, 'MLP Head\n[2048,1024]', '#E2D4F0'),
    (90, 50, 10, 'Output\n15 classes', '#D1E7DD'),
]

for x, y, w, text, color in boxes:
    rect = plt.Rectangle((x-w/2, y-10), w, 20, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows
for i in range(len(boxes)-1):
    ax.annotate('', xy=(boxes[i+1][0]-boxes[i+1][2]/2-1, boxes[i+1][1]),
                xytext=(boxes[i][0]+boxes[i][2]/2+1, boxes[i][1]),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Add title
ax.text(50, 85, 'Vision Transformer Architecture for Chest X-ray Classification', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Add details
details = """
Configuration:
• Patch Size: 32×32
• Num Patches: 49 (7×7)
• Projection Dim: 64
• Attention Heads: 4
• Transformer Layers: 8
• Total Parameters: ~9.4M
"""
ax.text(5, 20, details, ha='left', va='top', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig(os.path.join(output_dir, '02_vit_architecture.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: {os.path.join(output_dir, '02_vit_architecture.png')}")

# 3. Multi-label distribution
print("[10.3] Creating Multi-label Distribution Chart...")
fig, ax = plt.subplots(figsize=(10, 6))
label_hist = label_counts_per_image.value_counts().sort_index()
ax.bar(label_hist.index, label_hist.values, color='steelblue', edgecolor='black')
ax.set_xlabel('Number of Labels per Image', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Labels per Image', fontsize=14, fontweight='bold')
ax.set_xticks(range(1, max(label_counts_per_image)+1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_multilabel_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: {os.path.join(output_dir, '03_multilabel_distribution.png')}")

# 4. Quality Assessment Radar Chart
print("[10.4] Creating Quality Assessment Radar Chart...")
categories = ['Correctness', 'Performance', 'Robustness', 'Maintainability', 'Scalability', 'Reproducibility']
scores = [6.5, 6.75, 5.25, 5.0, 4.5, 5.5]

angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
scores_plot = scores + scores[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
ax.plot(angles, scores_plot, 'o-', linewidth=2, color='steelblue')
ax.fill(angles, scores_plot, alpha=0.25, color='steelblue')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_title('Quality Assessment Overview\n(Scale: 0-10)', fontsize=14, fontweight='bold', y=1.08)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_quality_radar.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: {os.path.join(output_dir, '04_quality_radar.png')}")

# 5. Sample images visualization
print("[10.5] Creating Sample Images Grid...")
if len(image_files) >= 9:
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    sample_imgs = random.sample(image_files, 9)
    
    for i, (ax, img_path) in enumerate(zip(axes.flat, sample_imgs)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        
        # Get label
        img_name = os.path.basename(img_path)
        label = df_available[df_available['Image Index'] == img_name]['Finding Labels'].values
        if len(label) > 0:
            label_text = label[0][:30] + '...' if len(label[0]) > 30 else label[0]
            ax.set_title(label_text, fontsize=9)
        ax.axis('off')
    
    plt.suptitle('Sample Chest X-ray Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_sample_images.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {os.path.join(output_dir, '05_sample_images.png')}")

# ============================================================================
# SECTION 11: FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SECTION 11: EXECUTIVE SUMMARY")
print("="*80)

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTIVE SUMMARY                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Project: Vision Transformer for Chest X-ray Disease Classification          │
│  Dataset: NIH Chest X-ray (Partial - """ + f"{len(image_files):,}" + """ images available)                   │
│  Primary Model: ViT with 9.4M parameters                                     │
│                                                                              │
│  OVERALL ASSESSMENT: """ + "⭐⭐⭐" + """ (3/5 stars)                                          │
│                                                                              │
│  Key Findings:                                                               │
│  ✓ Correct ViT implementation with proper attention mechanism                │
│  ✓ Multiple model variants for comparison (CNN, ResNet, ViT, Pre-trained)   │
│  ✓ Working end-to-end pipeline from data loading to evaluation              │
│  ✗ Critical class imbalance issue not addressed                             │
│  ✗ Code quality issues (missing imports, deprecated APIs)                   │
│  ✗ Limited evaluation metrics for medical imaging                           │
│                                                                              │
│  Recommendation: Address class imbalance and code issues before deployment  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
""")

print(f"\n[Analysis Complete] Output saved to: {output_dir}")
print(f"Generated files:")
for f in sorted(os.listdir(output_dir)):
    print(f"   • {f}")

print("\n" + "="*80)
print("END OF COMPREHENSIVE ANALYSIS REPORT")
print("="*80)
