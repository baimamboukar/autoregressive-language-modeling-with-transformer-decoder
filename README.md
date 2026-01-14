# Autoregressive Language Modeling with a Causal Transformer Decoder

_This project implements a comprehensive deep learning codebase for sequence modeling, featuring an **Autoregressive Decoder-only Transformer** for causal language modeling and an **Encoder-Decoder Transformer** for Automatic Speech Recognition (ASR). Built from the ground up using PyTorch and a custom `mytorch` library, the system provides an end-to-end pipeline for training, evaluation, and efficient decoding._


## Tools and Technologies

<p align="left">
  <a>
    <img src="https://skillicons.dev/icons?i=pytorch,python,github,githubactions,markdown,bash,stackoverflow&perline=19" alt="Skill Icons">
  </a>
  </p>


## Overview

* **Purpose:** The primary objective of this project is to implement, train, and optimize Transformer-based architectures from scratch. It aims to bridge the gap between theoretical understanding and practical implementation of self-attention mechanisms, positional encodings, and sequence generation strategies.
* **Model:**
  * **Autoregressive Decoder-Only Transformer:** A GPT-style architecture designed for causal language modeling, predicting the next token in a sequence based on previous context.
  * **Encoder-Decoder Transformer:** A complete Transformer architecture optimized for mapping audio features (speech) to text sequences (ASR).
* **Functionality:**
  * **Custom Library:** A modular `lib` and `mytorch` framework containing manual implementations of Multi-Head Attention, Linear layers, and Optimizers.
  * **Text Generation:** efficient text generation using Greedy and Beam Search decoding strategies.
  * **Speech Processing:** End-to-end speech-to-text pipeline with feature extraction and tokenization.
  * **Configurable Training:** Flexible experiment management using YAML configurations.

## Evaluation

* **Metrics:**

  * **Perplexity (PPL):** Used to evaluate the Autoregressive Language Model (P1).
  * **Character Error Rate (CER):** The primary metric for evaluating ASR performance (P2).
  * **Word Error Rate (WER):** Secondary metric for speech transcription accuracy.
