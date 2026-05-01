# Dual-Path RWKV-like EEG Encoder for Brain-Assisted Target Speaker Extraction

This repository provides the PyTorch implementation for anonymous review.

## Introduction

Brain-assisted target speaker extraction aims to selectively extract the target speaker's speech by decoding the listener's brain activity from electroencephalogram (EEG) signals. Compared with conventional audio-only target speaker extraction methods, EEG-guided systems provide additional neural information related to auditory attention, making it possible to identify which speaker the listener is attending to in a multi-speaker scenario.

However, EEG signals are usually characterized by low signal-to-noise ratio, high dimensionality, and non-stationarity. These properties make it challenging to efficiently learn discriminative and high-level EEG representations. In particular, modeling long EEG sequences and weak correlations across electrodes remains difficult for traditional EEG encoders.

To address these limitations, we propose a dual-path RWKV-like EEG encoder for brain-assisted target speaker extraction. The proposed encoder introduces cortical interaction and temporal dynamics modules to capture multidimensional spatiotemporal information from EEG signals with linear computational complexity.

## Method Overview

The proposed model contains a dual-path EEG encoder designed for efficient EEG representation learning.

The encoder mainly consists of:

- **Cortical Interaction Module**, which models cross-electrode dependencies and captures spatial interactions among EEG channels.
- **Temporal Dynamics Module**, which captures temporal variations and local dynamic patterns within EEG signals.
- **RWKV-like bidirectional weighted key-value mechanism**, which enables efficient long-sequence modeling with linear computational complexity.
- **Brain-assisted target speaker extraction framework**, which uses EEG representations to guide the extraction of the attended speaker from speech mixtures.

By combining spatial interaction and temporal dynamics modeling, the proposed encoder can effectively learn robust EEG features for target speaker extraction.



## Environment

The code is implemented with PyTorch. We recommend creating a clean Python environment before running the experiments.

```bash
conda create -n dp_eeg_tse python=3.10
conda activate dp_eeg_tse
pip install -r requirements.txt
```

## Train

To train the proposed model, run:

python distributed.py -c configs/DP_EEG_TSE.json

## Test

To evaluate the trained model, run:

python test.py -c configs/experiments.json