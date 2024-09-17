import os
from transformers import TrainingArguments, EarlyStoppingCallback

class Config:
    # Data
    DEBUG = False
    LOAD_PROCESSED_DATASET = True
    OUTPUT_DIR = "results"
    RAW_DATA_DIR = "dataset"
    NUM_VALUE_BINS = 10  # number of quantiles to bin float values to int tokens
    NUM_ADDED_TOKENS = 4  # {"pad": 0, "mask"/"unk": 1, "bos": 2, "eos": 3}
    
    # Training
    CALLBACKS = [
        EarlyStoppingCallback(
            early_stopping_patience=4,
            early_stopping_threshold=0.01,
        ),
    ]
    TRAINING_ARGUMENTS = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,
        bf16=True,
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1000 if not DEBUG else 1,
        load_best_model_at_end=True,
        metric_for_best_model="monitored_metric",
        greater_is_better=True,
    )
    
    # Model
    LM_TYPE = "masked"  # "masked", "causal"
    LM_PRETRAINING = "general"  # "general", "healthcare"
    LM_ID = 0  # 0, 1, 2
    POSSIBLE_MODELS = {
        "masked": {
            "general": [
                "bert-base-uncased",
                "roberta-base",
                "distilbert-base-uncased"
            ],
            "healthcare": [
                "emilyalsentzer/Bio_ClinicalBERT",
                "dmis-lab/biobert-base-cased-v1.1",
                "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
            ]
        },
        "causal": {
            "general": [
                "gpt2",
                "distilgpt2",
                "EleutherAI/gpt-neo-2.7B"
            ],
            "healthcare": [
                "microsoft/BioGPT",
                "lucadiliello/clinical-gpt",
                "stanford-crfm/pubmedgpt"
            ]
        }
    }
    ORIGINAL_MODEL_ID = POSSIBLE_MODELS[LM_TYPE][LM_PRETRAINING][LM_ID]