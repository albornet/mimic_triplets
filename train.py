import numpy as np
from sklearn.metrics import classification_report
from config import Config as cfg
from pprint import pprint
from transformers import Trainer
from model_utils import PatientMLMModel
from data_utils import PatientDataCollatorForLanguageModelling, create_patient_dataset


def main():
    """ Train a patient embedding model with MLM (or causal LM?)
    """
    # Initialize dataset
    dataset, feature_types_vocab, bin_edges_by_type = create_patient_dataset(
        raw_data_dir=cfg.RAW_DATA_DIR,
        load_processed_dataset=cfg.LOAD_PROCESSED_DATASET,
        num_value_bins=cfg.NUM_VALUE_BINS,
        num_added_tokens=cfg.NUM_ADDED_TOKENS,
        debug=cfg.DEBUG,
    )
    
    # Initialize model
    num_value_tokens = cfg.NUM_VALUE_BINS + cfg.NUM_ADDED_TOKENS
    num_type_tokens = len(feature_types_vocab) + cfg.NUM_ADDED_TOKENS
    model = PatientMLMModel(
        original_model_id=cfg.ORIGINAL_MODEL_ID,
        language_model_type=cfg.LM_TYPE,
        num_value_tokens=num_value_tokens,
        num_type_tokens=num_type_tokens,
    )
    
    # Initialize data collator, which also implements MLM for "values" field
    data_collator = PatientDataCollatorForLanguageModelling(
        mlm=True if cfg.LM_TYPE == "masked" else False,
        num_mlm_labels=cfg.NUM_VALUE_BINS,
        num_tokens_max=model.num_tokens_max,
    )
    
    # Define trainer object
    trainer = Trainer(
        model=model,
        args=cfg.TRAINING_ARGUMENTS,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=cfg.CALLBACKS,
    )
    
    # Train the model
    trainer.train()
    # save_model(model, os.path.join(cfg.RESULT_DIR, "model.safetensors"))
    
    # Evaluate the model
    results = trainer.evaluate(eval_dataset=dataset["test"])
    print("Test evaluation results:")
    pprint(results)


def compute_metrics_fn(eval_pred):
    """ Function for evaluating model trained for MLM
    """
    # Extract and flatten 
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Filer out all tokens where label is -100 (coding for padded tokens)
    mask = labels != -100
    filtered_predictions = predictions[mask]
    filtered_labels = labels[mask]
    
    # Compute relevant metrics
    report = classification_report(
        y_true=filtered_labels,
        y_pred=filtered_predictions,
        output_dict=True,
    )
    
    # Label the metric of interest
    report["monitored_metric"] = report["macro avg"]["f1-score"]
    
    return report
    

if __name__ == "__main__":
    main()