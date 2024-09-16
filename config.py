import os
from transformers import TrainingArguments, EarlyStoppingCallback

class Config:
    DEBUG = False
    ORIGINAL_MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    RESULT_DIR = "results"
    RAW_DATA_DIR = "dataset"
    LOAD_PROCESSED_DATASET = True
    NUM_VALUE_BINS = 10
    CALLBACKS = [EarlyStoppingCallback(early_stopping_patience=4)]
    TRAINING_ARGUMENTS = TrainingArguments(
        output_dir=RESULT_DIR,
        logging_dir=os.path.join(RESULT_DIR, "logs"),
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,
        bf16=True,
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=1000 if not DEBUG else 1,
        load_best_model_at_end=True,
        metric_for_best_model="monitored_metric",
        greater_is_better=True,
    )