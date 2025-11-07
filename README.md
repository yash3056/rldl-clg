# Deep Learning Experiments Summary

## Overview
This repository contains implementations of 10 deep learning experiments covering fundamental concepts from reinforcement learning to advanced neural network architectures.

## Environment Setup
```bash
# Initialize environment with uv
uv sync
```

## Experiments List

### Experiment 2: Q-Learning for CartPole
- **File**: `exp2.py`
- **Aim**: Implement Q-Learning algorithm
- **Key Concepts**: Reinforcement Learning, Q-Table, Epsilon-Greedy

### Experiment 3: Deep Q-Network (DQN)
- **File**: `exp3.py`
- **Aim**: Implement DQN with experience replay
- **Key Concepts**: Deep RL, Neural Networks, Replay Buffer

### Experiment 4: Policy Iteration
- **File**: `exp4.py`
- **Aim**: Implement Policy Iteration algorithm
- **Key Concepts**: MDP, Dynamic Programming, Value Iteration

### Experiment 5: [Previous Experiment]
- **File**: `exp5.py`, `exp5.md`

### Experiment 6: MNIST CNN Classification ✓
- **Files**: `exp6.py`, `exp6.md`
- **Aim**: Image classification on MNIST dataset using CNN
- **Result**: 99.16% test accuracy
- **Key Concepts**: Convolutional Neural Networks, Pooling, Fully Connected Layers
- **Run**: `CUDA_VISIBLE_DEVICES="" uv run python exp6.py`

### Experiment 7: Sentiment Analysis with RNN
- **Files**: `exp7.py`, `exp7.md`
- **Aim**: Train sentiment analysis on IMDB dataset using LSTM/GRU
- **Key Concepts**: Recurrent Neural Networks, LSTM, GRU, Text Classification
- **Run**: `CUDA_VISIBLE_DEVICES="" uv run python exp7.py`

### Experiment 8: NLP with Cornell Movie Dialogs
- **Files**: `exp8.py`, `exp8.md`
- **Aim**: Apply DL models in NLP using movie dialogue corpus
- **Key Concepts**: Text Processing, Tokenization, Embeddings, Dialogue Analysis
- **Run**: `CUDA_VISIBLE_DEVICES="" uv run python exp8.py`

### Experiment 9: Bidirectional RNNs
- **Files**: `exp9.py`, `exp9.md`
- **Aim**: Implement Bi-GRU and Bi-LSTM for prediction
- **Key Concepts**: Bidirectional Processing, Context Awareness, Sequential Models
- **Run**: `CUDA_VISIBLE_DEVICES="" uv run python exp9.py`

### Experiment 10: Multimodal Deep Learning
- **Files**: `exp10.py`, `exp10.md`
- **Aim**: Implement DL on multimodal dataset (image + text)
- **Key Concepts**: Multimodal Fusion, Multiple Input Branches, Feature Integration
- **Run**: `CUDA_VISIBLE_DEVICES="" uv run python exp10.py`

## Running Experiments

### Individual Experiment
```bash
# Run on CPU (recommended for compatibility)
CUDA_VISIBLE_DEVICES="" uv run python exp6.py
CUDA_VISIBLE_DEVICES="" uv run python exp7.py
CUDA_VISIBLE_DEVICES="" uv run python exp8.py
CUDA_VISIBLE_DEVICES="" uv run python exp9.py
CUDA_VISIBLE_DEVICES="" uv run python exp10.py
```

### Run All Experiments
```bash
# Create a script to run all experiments sequentially
for exp in exp6 exp7 exp8 exp9 exp10; do
    echo "Running $exp..."
    CUDA_VISIBLE_DEVICES="" uv run python ${exp}.py
done
```

## Code Structure
- Each experiment has minimal, optimized code
- Separate `.md` files contain theory and explanation
- All experiments use TensorFlow/Keras for neural networks
- Code follows consistent structure for easy understanding

## Dependencies
All dependencies are managed through `uv` and include:
- TensorFlow 2.20.0
- Keras 3.11.3
- NumPy 2.3.2
- Gymnasium 1.2.0
- PyTorch 2.9.0
- And more (see `uv sync` output)

## Notes
- Use `CUDA_VISIBLE_DEVICES=""` to force CPU execution if GPU compatibility issues arise
- Experiments 6-10 are designed to be concise and educational
- Each experiment includes comprehensive theory in the corresponding `.md` file
