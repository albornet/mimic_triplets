import os
from transformers import EarlyStoppingCallback


class Config:
    """ Class containing parameters for building the dataset and training the models
    """
    # Dataset building
    DEBUG = False
    BUILD_PATIENT_CSV_FILES = True
    BUILD_HUGGINGFACE_DATASET = False
    RESULT_DIR = "results"
    RAW_DATA_DIR = "raw_data"
    RAW_DATA_SUBSET = "eicu"
    PROCESSED_DATA_DIR = "processed_data"
    PRIMARY_KEY_MAP = {
        "eicu": "patientunitstayid",
        "eicu_demo": "patientunitstayid",
        "miiv": "stay_id",
        "mimic": "icustay_id",
        "mimic_demo": "icustay_id",
        "hirid": "patientid",
        "sic": "CaseID",
    }
    TIME_VARS_MAP_DATASET_LIST_KEYS  = {
        "eicu": [
            "infusionoffset", "labresultoffset", "nursingchartoffset",
            "observationoffset", "respchartoffset", "respcarestatusoffset",
            "intakeoutputoffset", "culturetakenoffset", "drugstartoffset", "index_var",
        ],
        "eicu_demo": [
            "infusionoffset", "labresultoffset", "nursingchartoffset",
            "observationoffset", "respchartoffset", "respcarestatusoffset",
            "intakeoutputoffset", "culturetakenoffset", "drugstartoffset", "index_var",
        ],
        "miiv": ["starttime", "charttime", "index_var", "chartdate"],
        "mimic": ["charttime", "chartdate", "index_var", "startdate", "starttime"],
        "mimic_demo": ["charttime", "chartdate", "index_var", "startdate", "starttime"],
        "hirid": ["givenat", "datetime", "index_var"],
        "sic": ["OffsetOfDeath", "Offset", "index_var"],
    }
    
    # Training arguments
    DEFAULT_NUM_VALUE_BINS = 10  # default value for number of quantiles to bin float values to int tokens
    TUNE_NUM_VALUE_BINS = False  # with optuna
    NUM_ADDED_TOKENS = 4  # {"pad": 0, "mask"/"unk": 1, "bos": 2, "eos": 3}
    NUM_TRAIN_EPOCHS = 1000
    DEFAULT_TRAINING_ARGUMENTS = dict(
        output_dir=RESULT_DIR,
        logging_dir=os.path.join(RESULT_DIR, "logs"),
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        warmup_steps=100,
        learning_rate=5e-5,
        bf16=True,
        weight_decay=0.01,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=NUM_TRAIN_EPOCHS if not DEBUG else 1,
        load_best_model_at_end=True,
        metric_for_best_model="monitored_metric",
        greater_is_better=True,
    )
    
    # Training utilities
    USE_OPTUNA = True
    NUM_OPTUNA_TRIALS = 100
    TRAINER_CALLBACKS = [
        EarlyStoppingCallback(
            early_stopping_patience=4,
            early_stopping_threshold=0.01,
        ),
    ]
    
    # Model
    LM_TYPE = "masked"  # "masked", "causal"
    LM_PRETRAINING = "healthcare"  # "general", "healthcare"
    LM_ID = 0  # 0, 1, 2
    POSSIBLE_MODELS = {
        "masked": {
            "general": [
                "bert-base-uncased",
                "roberta-base",
                "distilbert-base-uncased",
            ],
            "healthcare": [
                "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                "emilyalsentzer/Bio_ClinicalBERT",
                "dmis-lab/biobert-base-cased-v1.1",
            ]
        },
        "causal": {
            "general": [
                "gpt2",
                "distilgpt2",
                "EleutherAI/gpt-neo-2.7B",
            ],
            "healthcare": [
                "microsoft/BioGPT",
                "lucadiliello/clinical-gpt",
                "stanford-crfm/pubmedgpt",
            ]
        }
    }
    ORIGINAL_MODEL_ID = POSSIBLE_MODELS[LM_TYPE][LM_PRETRAINING][LM_ID]