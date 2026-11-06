
# ChromaFormer — So2Sat-LCZ42 Project

## 1. Project Overview

**Model:** ChromaFormer  
**Implementation:** PyTorch  
**Task:** Land Use / Land Cover Classification  
**Dataset:** So2Sat-LCZ42  
**Number of classes:** 17 LCZ classes

This project implements a ChromaFormer-inspired architecture adapted for Sentinel-1 and Sentinel-2 satellite imagery.

Original research paper:

ChromaFormer: A Scalable Multi-Spectral Transformer for Large-Scale Land Cover Classification.

https://openreview.net/pdf?id=qzJVTJYEBc

## 2. Original Architecture

The original ChromaFormer architecture combines:

- Swin Transformer backbone
- Spectral Dependency Module (SDM)
- Hierarchical feature extraction
- Multi-spectral feature learning

The architecture contains four hierarchical stages.

The ChromaFormer-T configuration uses:

| Parameter | Value |
|---|---:|
| Embedding dimension | 96 |
| Stage depths | 2, 2, 6, 2 |
| Attention heads | 3, 6, 12, 24 |
| Approximate original parameters | 27 million |

## 3. Spectral Dependency Module (SDM)

The SDM learns relationships between feature channels using channel-to-channel attention.

The input feature map is reshaped into a sequence of spatial features.

Attention is calculated using:

A = softmax(QK^T / sqrt(N))

where N represents the number of spatial positions.

The attention output is then reshaped into the original feature-map dimensions.

## 4. Sentinel-1 and Sentinel-2 Adaptation

The So2Sat-LCZ42 dataset contains:

- Sentinel-1: 8 channels
- Sentinel-2: 10 channels

The implementation supports three input configurations:

| Input | Channels |
|---|---:|
| Sentinel-1 only | 8 |
| Sentinel-2 only | 10 |
| Combined Sentinel-1 + Sentinel-2 | 18 |

For combined input, the two sensor inputs are concatenated along the channel dimension.

The input projection layer is configurable to accommodate these different channel counts.

This is an adaptation for the So2Sat-LCZ42 dataset, not the original paper's exact input configuration.

## 5. Model Architecture Components

The implementation contains:

1. Spectral Dependency Module
2. Swin Transformer feature block
3. Patch embedding
4. Patch merging
5. Four hierarchical feature stages
6. Global average pooling
7. 17-class classification head

The original paper uses a dense land-cover segmentation task. This implementation adapts the architecture for patch-level LCZ classification.

## 6. Input and Output Dimensions

Combined sensor input:

(batch_size, 18, 32, 32)

Sentinel-2-only input:

(batch_size, 10, 32, 32)

Sentinel-1-only input:

(batch_size, 8, 32, 32)

Model output:

(batch_size, 17)

Each output contains 17 class scores corresponding to the 17 LCZ classes.

## 7. Implementation and Testing

The implementation was developed using PyTorch.

Individual components were tested for tensor dimension compatibility.

The complete model forward pass was tested successfully.

Combined sensor test:

Input: (2, 18, 32, 32)
Output: (2, 17)

Sentinel-2 test:

Input: (2, 10, 32, 32)
Output: (2, 17)

Sentinel-1 test:

Input: (2, 8, 32, 32)
Output: (2, 17)

All three configurations produced the expected output dimensions.

## 8. Computational Requirements

Measured implementation results:

| Measurement | Result |
|---|---:|
| Total parameters | 27,537,375 |
| Trainable parameters | 27,537,375 |
| FP32 parameter memory | 105.05 MB |
| Forward-pass time | 0.1574 seconds |
| Test batch size | 2 |
| Input size | 32 × 32 |
| Output classes | 17 |

Environment:

- PyTorch: 2.15.0+cpu
- Torchvision: 0.29.2+cpu
- Execution device: CPU

The parameter memory measurement represents model weights only. Actual runtime memory also includes activations and other allocations.

The forward-pass time is a single initial measurement, not a formal benchmark.

## 9. Limitations and Future Work

- Verify all architectural details against the original ChromaFormer implementation.
- Confirm the exact SDM formulation and placement.
- Validate the model using real Sentinel-1 and Sentinel-2 samples.
- Verify sensor alignment and normalization.
- Evaluate classification accuracy using validation data.
- Measure peak runtime memory and averaged inference time.
- Compare results against other classification models.

## 10. Deliverables

- chromaformer.py — PyTorch model implementation.
- chromaformer_documentation.md — Model documentation.
- Forward-pass test — Successful output dimension verification.

**Note:** This is a ChromaFormer-inspired adaptation for patch-level LCZ classification. It has passed preliminary forward-pass tests, but has not yet been established as an exact reproduction of the published architecture.
