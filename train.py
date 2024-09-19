import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import optuna
from pprint import pprint
from config import Config as cfg
from transformers import Trainer, TrainingArguments, TrainerCallback
from model_utils import (
    PatientEmbeddingModel,
    PatientDataCollatorForLanguageModelling,
)
from data_utils import create_patient_dataset
from sklearn.metrics import classification_report


def main():
    """ Train a patient embedding model with MLM (or causal LM?)
    """
    # Initialize dataset, model, and associated data collator
    dataset, model, data_collator = init_patient_embedding_run()
    
    # Define or look for the best training arguments
    training_arguments = TrainingArguments(**cfg.DEFAULT_TRAINING_ARGUMENTS)
    if cfg.FIND_BEST_TRAINING_ARGUMENTS_WITH_OPTUNA:
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(optuna_objective_fn, n_trials=cfg.NUM_OPTUNA_TRIALS)
        updated_training_arguments = training_arguments.to_dict()
        updated_training_arguments.update(**study.best_params)
        training_arguments = TrainingArguments(**updated_training_arguments)
    
    # Define trainer object
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=cfg.TRAINER_CALLBACKS,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    results = trainer.evaluate(eval_dataset=dataset["test"])
    print("Test evaluation results:")
    pprint(results)


def compute_metrics_fn(eval_pred):
    """ Function for evaluating model trained for MLM or CausalLM
        TODO: CHANGE FOR CLUSTERING CSS LABEL METRIC!
    """
    # Extract prediction and labels
    logits, labels = eval_pred
    if isinstance(logits, tuple): logits = logits[0]  # in case LM returns more
    predictions = np.argmax(logits, axis=-1)
    
    # Flatten predictions and labels for evaluation
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Filter out all tokens where label is -100 (coding for padded tokens)
    mask = labels != -100
    filtered_predictions = predictions[mask]
    filtered_labels = labels[mask]
    
    # Compute relevant metrics
    report = classification_report(
        y_true=filtered_labels,
        y_pred=filtered_predictions,
        output_dict=True,
        zero_division=0,
    )
    
    # Label the metric of interest for optional stopping
    report["monitored_metric"] = report["macro avg"]["f1-score"]
    
    return report


def init_patient_embedding_run():
    """ Initialize dataset, model and data collator for patient embedding
        TODO: INCLUDE NUM_VALUE_BINS AS A HYPER-PARAMETER OPTIMIZED WITH OPTUNA,
        BUT THIS WOULD REQUIRE TO HAVE ANOTHER METRIC FUNCTION, E.G., CLUSTERING
    """
    # Initialize dataset
    dataset, feature_types_vocab, bin_edges_by_type = create_patient_dataset(
        raw_data_dir=cfg.RAW_DATA_DIR,
        load_processed_dataset=cfg.LOAD_PROCESSED_DATASET,
        num_value_bins=cfg.NUM_VALUE_BINS,
        num_added_tokens=cfg.NUM_ADDED_TOKENS,
        debug=cfg.DEBUG,
    )
    
    # Initialize model with correct LM type and number of embedded features
    num_value_tokens = cfg.NUM_VALUE_BINS + cfg.NUM_ADDED_TOKENS
    num_type_tokens = len(feature_types_vocab) + cfg.NUM_ADDED_TOKENS
    model = PatientEmbeddingModel(
        original_model_id=cfg.ORIGINAL_MODEL_ID,
        language_model_type=cfg.LM_TYPE,
        num_value_tokens=num_value_tokens,
        num_type_tokens=num_type_tokens,
    )
    
    # Initialize data collator, which implements MLM or CausalLM
    data_collator = PatientDataCollatorForLanguageModelling(
        mlm=True if cfg.LM_TYPE == "masked" else False,
        num_mlm_labels=cfg.NUM_VALUE_BINS,
        num_tokens_max=model.num_tokens_max,
    )
    
    return dataset, model, data_collator


def optuna_objective_fn(trial: optuna.Trial) -> float:
    """ Find the best set of hyper-parameters to train a patient embedding model
        TODO: DEFINE HYPER-PARAMETER RANGES AS A DICT IN THE CONFIG FILE
    """
    # Hyperparameters to optimize
    new_training_parameters = {
        "per_device_train_batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "disable_tqdm": True,  # to avoid clutter during Optuna trials
    }
    
    # Initialize new patient embedding model and data collator
    dataset, model, data_collator = init_patient_embedding_run()
    
    # Define training arguments
    training_arguments = dict(cfg.DEFAULT_TRAINING_ARGUMENTS)
    training_arguments.update(**new_training_parameters)
    training_arguments = TrainingArguments(**training_arguments)
    
    # Define trainer, with pruning callback (pruning not working for now)
    callbacks = cfg.TRAINER_CALLBACKS + [OptunaCallback(trial)]
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
    
    # Train the model (may be pruned along the way!)
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    
    # Return the metric of interest (TODO: CHANGE TO CLUSTERING CSS LABEL METRIC)
    return eval_results["eval_monitored_metric"]


class OptunaCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        
    def on_evaluate(self, args, state, control, **kwargs):
        eval_metrics = kwargs.get("metrics", {})
        monitored_metric = eval_metrics.get("eval_monitored_metric")
        
        if monitored_metric is not None:
            self.trial.report(monitored_metric, step=state.global_step)
        
        if self.trial.should_prune():
            raise optuna.TrialPruned()


if __name__ == "__main__":
    main()