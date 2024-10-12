import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import optuna
from pprint import pprint
from config import Config as cfg
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, TrainerCallback
from model_utils import (
    PatientEmbeddingModel,
    PatientDataCollatorForLanguageModelling,
)
from data_utils import get_formatted_patient_dataset
from sklearn.metrics import classification_report


def main():
    """ Train a patient embedding model with MLM (or causal LM?)
    """
    # Define or look for the best training arguments
    num_value_bins, training_arguments = choose_hyper_parameters(cfg.USE_OPTUNA)
        
    # Initialize dataset, model, and associated data collator
    dataset, model, data_collator = init_patient_embedding_run(num_value_bins)
    
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


def choose_hyper_parameters(
    use_optuna: bool=False,
) -> tuple[int, TrainingArguments]:
    """ Select hyper-parameters (how value binning is done and trainig arguments)
        by running an optuna study or by selecting default values
    """
    # Default values
    num_value_bins = cfg.DEFAULT_NUM_VALUE_BINS
    training_arguments = TrainingArguments(**cfg.DEFAULT_TRAINING_ARGUMENTS)
    
    # Update default values with optuna study
    if use_optuna:
        sampler = optuna.samplers.TPESampler()
        storage = f"sqlite:///{os.path.join(cfg.RESULT_DIR, 'optuna.db')}"
        study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage)
        study.optimize(optuna_objective_fn, n_trials=cfg.NUM_OPTUNA_TRIALS)
        
        # Update value binning hyper-parameter
        if "num_value_bins" in study.best_params:
            num_value_bins = study.best_params.pop("num_value_bins")
        
        # Update training arguments
        updated_training_arguments = training_arguments.to_dict()
        updated_training_arguments.update(**study.best_params)
        training_arguments = TrainingArguments(**updated_training_arguments)
    
    return num_value_bins, training_arguments


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


def init_patient_embedding_run(
    num_value_bins: int,
) -> tuple[
    DatasetDict,
    PatientEmbeddingModel,
    PatientDataCollatorForLanguageModelling,
]:
    """ Initialize dataset, model and data collator for patient embedding
    """
    # Initialize dataset
    dataset, feature_types_vocab, bin_edges_by_type = get_formatted_patient_dataset(
        num_value_bins=num_value_bins,
        num_added_tokens=cfg.NUM_ADDED_TOKENS,
    )
    
    # Initialize model with correct LM type and number of embedded features
    num_value_tokens = num_value_bins + cfg.NUM_ADDED_TOKENS
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
        num_mlm_labels=num_value_bins,
        num_tokens_max=model.num_tokens_max,
    )
    
    return dataset, model, data_collator


def optuna_objective_fn(trial: optuna.Trial) -> float:
    """ Find the best set of hyper-parameters to train a patient embedding model
    """
    # Hyperparameters to optimize
    new_training_parameters = {
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16, 32, 64],
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "adam_beta1": trial.suggest_float("adam_beta1", 0.8, 0.99, log=True),
        "adam_beta2": trial.suggest_float("adam_beta2", 0.9, 0.999, log=True),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-8, 1e-6, log=True),
        "disable_tqdm": True,  # to avoid clutter during Optuna trials
    }
    
    # Initialize new patient embedding model and data collator
    if cfg.TUNE_NUM_VALUE_BINS:
        num_value_bins = trial.suggest_int("num_value_bins", 2, 100, log=True)
    else:
        num_value_bins = cfg.DEFAULT_NUM_VALUE_BINS
    dataset, model, data_collator = init_patient_embedding_run(num_value_bins)
    
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
    """ Optuna-specific callback added to check if a trial is promising during
        the run, compared to previous trials, and prune it if not
    """
    def __init__(self, trial: optuna.Trial):
        """ Record trial being checked
        """
        self.trial = trial
        
    def on_evaluate(self, args, state, control, **kwargs):
        """ When model is evaluated during training, report the metric to optuna,
            which checks if the trial should be pruned compared to previous ones
        """
        eval_metrics = kwargs.get("metrics", {})
        monitored_metric = eval_metrics.get("eval_monitored_metric")
        
        if monitored_metric is not None:
            self.trial.report(monitored_metric, step=state.global_step)
        
        if self.trial.should_prune():
            raise optuna.TrialPruned()


if __name__ == "__main__":
    main()