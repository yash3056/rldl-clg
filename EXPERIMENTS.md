# Experiments Quick Reference

| Exp # | File | Lines | Aim | Dataset | Model | Status |
|-------|------|-------|-----|---------|-------|--------|
| 6 | exp6.py | 26 | Image Classification | MNIST | CNN | ✓ Ready |
| 7 | exp7.py | 36 | Sentiment Analysis | IMDB | LSTM/GRU | ✓ Ready |
| 8 | exp8.py | 61 | NLP Application | Cornell Dialogs | RNN | ✓ Ready |
| 9 | exp9.py | 47 | Bidirectional RNN | IMDB | Bi-LSTM/Bi-GRU | ✓ Ready |
| 10 | exp10.py | 74 | Multimodal Learning | MNIST (Image+Text) | CNN+Dense | ✓ Ready |

## Quick Start

```bash
# Run individual experiment
CUDA_VISIBLE_DEVICES="" uv run python exp6.py

# Or run all at once
./run_all_experiments.sh
```

## File Structure
```
rldl-clg/
├── exp6.py & exp6.md     → CNN on MNIST
├── exp7.py & exp7.md     → RNN Sentiment Analysis
├── exp8.py & exp8.md     → NLP with Cornell Dialogs
├── exp9.py & exp9.md     → Bidirectional RNNs
├── exp10.py & exp10.md   → Multimodal DL
├── run_all_experiments.sh → Run all experiments
└── README.md             → Full documentation
```

## Theory Coverage

Each `.md` file includes:
- **Aim**: Clear objective statement
- **Theory**: Comprehensive explanation of concepts
  - Architecture details
  - Key components
  - Mathematical intuition
  - Practical applications

## Implementation Highlights

### Experiment 6 - MNIST CNN
- 2 Conv layers (32, 64 filters)
- MaxPooling for dimension reduction
- Dropout for regularization
- **Result**: ~99% accuracy

### Experiment 7 - Sentiment Analysis
- Compares LSTM vs GRU
- Embedding layer (10k vocab)
- Binary classification
- **Trains in**: ~3 epochs

### Experiment 8 - Cornell NLP
- Real dialogue corpus processing
- Text cleaning & tokenization
- LSTM-based text classifier
- Demonstrates end-to-end NLP pipeline

### Experiment 9 - Bidirectional Models
- Bi-LSTM and Bi-GRU comparison
- Captures forward & backward context
- Enhanced feature representation
- **Better accuracy** than unidirectional

### Experiment 10 - Multimodal
- Dual input branches (Image + Text)
- Feature fusion layer
- Demonstrates multimodal integration
- Shows improvement over single modality

## Code Philosophy
- **Minimal**: Each file under 75 lines
- **Complete**: Fully functional implementations
- **Clear**: Easy to understand structure
- **Educational**: Comments where needed
