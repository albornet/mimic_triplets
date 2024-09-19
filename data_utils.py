import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, Value, Sequence, concatenate_datasets
from functools import partial
from collections import defaultdict
from typing import Union
from tqdm import tqdm


def main():
    """ Test dataset building pipeline
    """
    from config import Config as cfg
    dataset, feature_type_vocab = create_patient_dataset(
        raw_data_dir=cfg.RAW_DATA_DIR,
        load_processed_dataset=cfg.LOAD_PROCESSED_DATASET,
        num_value_bins=cfg.NUM_VALUE_BINS,
        num_added_tokens=cfg.NUM_ADDED_TOKENS,
        debug=cfg.DEBUG
    )
    import ipdb; ipdb.set_trace()


def create_patient_dataset(
    raw_data_dir: str,
    load_processed_dataset: bool,
    num_value_bins: int,
    num_added_tokens: int,
    debug: bool=False,
) -> tuple[DatasetDict, dict[str, int], dict[str, np.ndarray]]:
    """ Load or create and save a huggingface patient dataset
    
    Returns:
        Dataset: processed huggingface dataset
    """
    # Define relevant data directories and paths
    preprocessed_data_dir = os.path.join(raw_data_dir, "input_features")
    processed_data_dir = os.path.join(raw_data_dir, "processed")
    processed_data_path = os.path.join(processed_data_dir, "huggingface_dataset")
    
    # Build dataset from raw feature csv files
    if load_processed_dataset == False:
        write_patient_triplet_csvs(raw_data_dir, preprocessed_data_dir, debug)
        dataset = build_patient_dataset(preprocessed_data_dir)
        dataset.save_to_disk(processed_data_path)
    
    # Load dataset from previously saved file
    else:
        dataset = DatasetDict.load_from_disk(processed_data_path)
    
    # Identify vocabulary for "types" feature set (note: not using test set)
    all_types = set([
        t for split, data in dataset.items() if split in ["train", "validation"]
        for sample in data for t in sample["types"]
    ])
    type_vocab = {k: v + num_added_tokens for v, k in enumerate(all_types)}
    
    # Format dataset to huggingface format, but with input_embed as input
    dataset, bin_edges_by_type = bin_values_by_type(dataset, num_value_bins, num_added_tokens)
    dataset = dataset.map(partial(encode_fn, type_vocab=type_vocab))
    dataset.set_format(type="torch", columns=["times", "values", "types"])
    
    return dataset, type_vocab, bin_edges_by_type


def bin_values_by_type(
    dataset: DatasetDict,
    num_value_bins: int,
    num_added_tokens: int,
) -> DatasetDict:
    """ Post-processing a huggingface dataset dictionary to bin values by
        quantiles computed over each feature type
    """
    # Use only training and validation to compute quantile bins
    train_val_data = concatenate_datasets([dataset["train"], dataset["validation"]])
    
    # Group values by type with all dataset samples
    values_by_type = defaultdict(list)
    for sample in train_val_data:
        values = sample["values"]  # list of floats
        types = sample["types"]  # list of categories (str)
        for value, type_ in zip(values, types):
            values_by_type[type_].append(value)
    
    # Compute N-tile bin edges for each type. In this conext, "100" means "100%",
    # and "-1" is used because "right=True" is used when digitizeing, which means
    # very large values will be assigned the last bin
    binned_space = np.linspace(0, 100, num_value_bins - 1)
    bin_edges_by_type = {}
    for type_, values in values_by_type.items():
        bin_edges_by_type[type_] = np.percentile(values, binned_space)
    
    # Bin the values for each sample, i.e., map each float-value to an int
    def bin_sample_values(times, values, types):
        """ Find value bin index using the precomputed bin edges for a given type
        """
        binned_values = []
        for value, type_ in zip(values, types):
            bin_edges = bin_edges_by_type[type_]  # defined outside function space
            bin_index = np.digitize(value, bin_edges, right=True)  # intervals include right bin_edges
            binned_values.append(bin_index + num_added_tokens)
        
        return {"times": times, "values": binned_values, "types": types}
    
    # Apply the binning to all dataset samples (including test set)
    bin_fn = lambda s: bin_sample_values(s["times"], s["values"], s["types"])
    binned_dataset = dataset.map(bin_fn)
    
    # Update data type for the "values" field (from float to int)
    original_features = binned_dataset["train"].features
    new_feature = Features({"values": Sequence(Value("int64"))})
    updated_features = Features({**original_features, **new_feature})
    binned_dataset = binned_dataset.cast(updated_features)
    
    return binned_dataset, bin_edges_by_type


def encode_fn(
    sample: dict[str, list[Union[float, str]]],
    type_vocab: dict[str, int],
) -> dict[str, torch.Tensor]:
    """ Tokenize and tensorize a formated patient triplet list
    """
    # Tensorize and add feature dimension to times
    sample["times"] = torch.tensor(sample["times"], dtype=torch.float32)
    sample["times"] = sample["times"].unsqueeze(-1)
    
    # Tensorize values
    sample["values"] = torch.tensor(sample["values"], dtype=torch.int64)
    
    # Tensorize and encode types, using 1 for unknown feature type tokens
    sample["types"] = [type_vocab.get(type_str, 1) for type_str in sample["types"]]
    sample["types"] = torch.tensor(sample["types"], dtype=torch.int64)
    
    return sample


def build_patient_dataset(preprocessed_data_dir: str) -> DatasetDict:
    """ Build a huggingface dataset from individual patient csv files
    
    Returns:
        Dataset: processed huggingface dataset
    """
    # Function to read a patient csv file to a dictionary
    def process_csv(file_path):
        df = pd.read_csv(file_path)
        return {k: df[k].tolist() for k in ["times", "values", "types"]}
        
    # Create list of patient dictionaries
    data = []
    for file_name in os.listdir(preprocessed_data_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(preprocessed_data_dir, file_name)
            data.append(process_csv(file_path))
    
    # Create train, validation, and test splits by patient (70% // 10% // 20%)
    train_data, valtest_data = train_test_split(data, test_size=0.3, random_state=1)
    val_data, test_data = train_test_split(valtest_data, test_size=0.667, random_state=1)
    
    # Return huggingface dataset dictionary that includes all splits
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })
    

def write_patient_triplet_csvs(
    raw_data_dir: str,
    processed_data_dir: str,
    debug: bool=False,
) -> list[str]:
    """
    Aggregate patient features from CSV files and record feature types

    Args:
        data_dir (str): path to the directory containing patient folders

    Returns:
        list[str]: unique feature types found across all patient folders
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    for patient_index, patient_folder in enumerate(tqdm(
        os.listdir(raw_data_dir),
        "Building input features",
    )):
        # Check patient index and initialize patient folder
        if debug == True and patient_index > 10: break
        if len(patient_folder) != 6 or not patient_folder.isdigit(): continue
        full_patient_folder = os.path.join(raw_data_dir, patient_folder)
        
        # Fill-in the data tuples with time-value-feature_type triplets
        data_tuples = []
        for filename in os.listdir(full_patient_folder):
            if filename.endswith(".csv") and filename != "icds.csv":
                file_path = os.path.join(full_patient_folder, filename)
                data_tuples.extend(extract_feature_triplets(file_path))
                
        # Sort the tuples by time (first element)
        data_tuples.sort(key=lambda x: x[0])
        
        # Save to new CSV file in the same folder
        output_path = os.path.join(processed_data_dir, "%s.csv" % patient_folder)
        df = pd.DataFrame(data_tuples, columns=["times", "values", "types"])
        df.to_csv(output_path, index=False)


def extract_feature_triplets(
    file_path: str,
) -> list[tuple[float, float, str]]:
    """ Extract a list of added time-value-feature_type triplet for a csv file
    """
    # Load data from feature csv file
    data = pd.read_csv(file_path)
    columns = data.columns
    
    # Initialize triplet feature vector list
    triplet_vector_list = []
    
    # Process time series data
    if "charttime" in columns or "startdate" in columns:
        if len(columns) > 3:
            data = data.iloc[:, [1, 2, 3]]
        else:
            data = data.iloc[:, [1, 2]]
        for _, row in data.iterrows():
            time = row.iloc[0]
            value = row.iloc[1]
            feature_type = columns[2]
            triplet_vector_list.append((time, value, feature_type))
    
    # Process static data
    else:
        for col in columns[1:]:
            time = 0  # pseudo time value for static data
            value = data[col].values[0]
            feature_type = col
            triplet_vector_list.append((time, value, feature_type))
            
    return triplet_vector_list


if __name__ == "__main__":
    main()
    