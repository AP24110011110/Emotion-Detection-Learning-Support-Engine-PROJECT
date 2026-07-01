# Machine Learning Training Guide

## Dataset Overview

This project uses emotion classification on educational learning problems.

### Recommended Datasets

**Option 1: Synthetic Dataset (Quick Start - Used by Default)**
- 1,000 samples (200 per emotion)
- Generated automatically by `train_bilstm_model.py`
- Covers 5 emotions: Bored, Confident, Confused, Curious, Frustrated
- Immediate use without external downloads
- Good for development and testing

**Option 2: Public Datasets**
- GoEmotions (58,000 Reddit comments, Google)
- ISEAR (7,666 emotion descriptions)
- Kaggle Emotion Dataset (20,000+ tweets)
- Custom educational datasets

Download format: CSV with columns `text` and `emotion`

## Training Pipeline

The training pipeline processes data in this order:

```
Raw Dataset (CSV)
    ↓
Load & Validate (training/dataset.py)
    ↓
Clean Text (normalize, tokenize)
    ↓
Build Tokenizer (fit on training set)
    ↓
Convert to Sequences (texts → integers)
    ↓
Pad Sequences (fixed length)
    ↓
Train/Test Split (80/20)
    ↓
BiLSTM Model Training
    ↓
Model Evaluation (on validation set)
    ↓
Save Model & Tokenizer
```

## Training the BiLSTM Model

### Step 1: Run the Training Script

```bash
cd "Emotion Detection & Learning Support Engine"
python train_bilstm_model.py
```

This will:
1. Generate synthetic dataset if `data/raw/emotion_data.csv` doesn't exist
2. Load and validate the dataset
3. Preprocess and tokenize text
4. Train the BiLSTM model for 12 epochs
5. Save model to `models/bilstm/bilstm_model.h5`
6. Save tokenizer to `models/tokenizer/bilstm_tokenizer.json`

### Step 2: Monitor Training

The script will print:
- Epoch progress (loss, accuracy)
- Validation metrics
- Early stopping triggers (if validation loss plateaus)
- File locations of saved artifacts

### Step 3: Verify Output Files

Check that these files exist after training completes:

```
models/
  bilstm/
    bilstm_model.h5          (← Your trained model)
    bilstm_model.checkpoint.h5 (backup)
  tokenizer/
    bilstm_tokenizer.json    (← Tokenizer for inference)
```

## Using a Custom Dataset

To use your own emotion dataset:

1. Create a CSV file with columns: `text`, `emotion`
2. Ensure emotions match the 5 classes: Bored, Confident, Confused, Curious, Frustrated
3. Save to `data/raw/emotion_data.csv`
4. Run: `python train_bilstm_model.py`

Example CSV format:
```
text,emotion
"I don't understand recursion at all",Confused
"I'm confident in my solution",Confident
"This is too easy and boring",Bored
"How does machine learning work?",Curious
"I'm stuck and frustrated",Frustrated
```

## Model Architecture

**BiLSTM Model:**
- Embedding layer (vocab_size × 128)
- SpatialDropout1D (30% dropout)
- Bidirectional LSTM (128 units)
- GlobalMaxPool1D (temporal dimension reduction)
- Dense layer (64 units, ReLU)
- Output layer (5 units, softmax)

**Training Hyperparameters:**
- Batch size: 32
- Learning rate: 2e-4
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Early stopping: patience=3 on validation loss
- Epochs: 12 (or until early stop)

## Training Time

On a typical machine:
- Synthetic dataset (1,000 samples): 2-3 minutes
- Public datasets (10,000+ samples): 10-20 minutes
- GPU acceleration: 3-5x faster

## Troubleshooting

**Issue: "Dataset CSV not found"**
- Solution: Script will auto-generate synthetic dataset

**Issue: Memory error during training**
- Solution: Reduce batch_size in train_bilstm_model.py (e.g., 16 instead of 32)

**Issue: Model not improving (high loss)**
- Solution: Increase epochs or check dataset quality

**Issue: Out of vocabulary words**
- Solution: Model uses <OOV> token, tokenizer handles unknown words automatically

## Next Steps

After training completes successfully:

1. Run the Streamlit app: `python -m streamlit run app.py`
2. The app will load your trained BiLSTM model automatically
3. Test predictions on student learning problems
4. (Optional) Fine-tune BERT model with: `python training/train_bert.py`

## Dataset Validation

The training script validates:
- Required columns (text, emotion)
- Valid emotion labels (must match constants.py)
- No missing or duplicate text
- Text normalization (lowercase, alphanumeric)
- Train/test split reproducibility (random_state=42)
