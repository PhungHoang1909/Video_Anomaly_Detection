## **Model Overview**

### **1. Sparse Feature Pyramid**
- Extracts multi-scale features from video frames using 3D convolution layers.
- Combines hierarchical feature maps with sparse lateral connections to retain spatial detail.

### **2. Swin Transformer Blocks**
- Employs window-based self-attention for efficient local and global spatial dependency modeling.
- Incorporates cyclic shift to improve feature diversity.

### **3. Temporal Transformer**
- Models temporal dependencies across video frames.
- Utilizes causal self-attention with a triangular mask to enforce temporal coherence.

### **4. Prompt Memory Bank**
- Stores normal patterns in a memory module for anomaly comparison.
- Enhances query features with learnable prompts to guide attention.

### **5. Decoder with Skip Connections**
- Reconstructs video frames using transposed convolutions and hierarchical upsampling.
- Integrates skip connections with adjusted feature dimensions to fuse multi-scale features.

## **Performance Metrics on Ped2 (at 35 epoches):**

- **Loss**: 2.7726
- **ROC AUC**: 0.6178
- **PR AUC**: 0.8819
