import os
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, Value, Sequence, concatenate_datasets
from functools import partial
from collections import defaultdict
from typing import Union
from tqdm import tqdm
from config import Config as cfg
from concurrent.futures import ProcessPoolExecutor


def main():
    """ Test dataset building pipeline
    """
    _ = create_patient_dataset()
    
    
def create_patient_dataset() -> str|None:
    """ Load or create and save a huggingface patient dataset after processing
        single patient feature csv files into combined patient csv files
        
        Return: (str) path to huggingface dataset dir, or None if not there
    """
    # Define relevant data directories and paths
    assert cfg.RAW_DATA_SUBSET in cfg.PRIMARY_KEY_MAP, "Invalid raw data subset"
    raw_data_full_dir = os.path.join(cfg.RAW_DATA_DIR, cfg.RAW_DATA_SUBSET)
    processed_data_full_dir = os.path.join(cfg.PROCESSED_DATA_DIR, cfg.RAW_DATA_SUBSET)
    patient_csv_data_dir = os.path.join(processed_data_full_dir, "patient_csvs")
    huggingface_data_dir = os.path.join(processed_data_full_dir, "huggingface")
    
    # Build dataset from raw feature csv files
    if cfg.BUILD_PATIENT_CSV_FILES:
        print("Building patient csv files from raw data directory")
        write_patient_triplet_csvs(raw_data_full_dir, patient_csv_data_dir, cfg.DEBUG)
    else:
        print("Assuming patient csv files are already built")
    
    # Huggingface dataset building / loading
    if cfg.BUILD_HUGGINGFACE_DATASET:
        print("Building huggingface dataset from patient csv files")
        dataset = build_huggingface_patient_dataset(patient_csv_data_dir)
        dataset.save_to_disk(huggingface_data_dir)
    else:
        print("Assuming huggingface dataset is already built or is not required")
    
    # Some return logic
    if os.path.exists(huggingface_data_dir)\
    and "dataset_dict.json" in os.listdir(huggingface_data_dir):
        return huggingface_data_dir
    else:
        print("No huggingface dataset was found at %s" % huggingface_data_dir)
        return None


def build_huggingface_patient_dataset(
    preprocessed_data_dir: str,
) -> DatasetDict:
    """ Build a huggingface dataset from individual patient csv files
    
    Returns:
        Dataset: processed huggingface dataset
    """
    # Function to read a patient csv file to a dictionary
    def process_csv(file_path):
        df = pd.read_csv(file_path)
        return {k: df[k].tolist() for k in ["times", "values", "types"]}
        
    # Create list of patient dictionaries read from patient csv files
    data = []
    for folder, _, file_names in os.walk(preprocessed_data_dir):  # recursive walk
        for file_name in file_names:
            file_root, file_ext = os.path.splitext(file_name)
            is_valid_file_name = (file_root.isdigit() and file_ext == ".csv")
            if not is_valid_file_name: continue
            file_path = os.path.join(folder, file_name)
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
    
    
def get_formatted_patient_dataset(
    num_value_bins: int,
    num_added_tokens: int,
) -> tuple[DatasetDict, dict[str, int], dict[str, np.ndarray]]:
    """ Create a patient dataset, then format it for language model processing
    
        Returns:
            dataset (DatasetDict): formatted huggingface patient dataset
            type_vocab (dict[str, int]): mapping of type tokens to type token ids
            bin_edges_by_type (dict[str, np.ndarray]): quantile bins for each type
    """
    # Create or load unformatted dataset
    dataset_path = create_patient_dataset(num_value_bins, num_added_tokens)
    dataset = DatasetDict.load_from_disk(dataset_path)
    
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
    

def write_patient_triplet_csvs(
    raw_data_dir: str,
    patient_csv_dir: str,
    debug: bool=False,
) -> list[str]:
    """
    Aggregate patient features from CSV files in parallel and record feature types.
    
    Args:
        raw_data_dir (str): path to the directory containing patient folders
        patient_csv_dir (str): path to the directory where processed CSVs will be saved
        debug (bool): if True, processes only the first 10 patients
    
    Returns:
        list[str]: list of processed patient folder names
    """
    # Checks for the directory where patient csvs are written
    if os.path.exists(patient_csv_dir) and os.listdir(patient_csv_dir):
        print(f"The directory {patient_csv_dir} is not empty! Still continuing.")
    os.makedirs(patient_csv_dir, exist_ok=True)
    
    # Get list of patient input features folders (raw input data)
    patient_features_folders = [
        os.path.join(raw_data_dir, f)
        for f in os.listdir(raw_data_dir) if f.isdigit()
    ]
    if debug: patient_features_folders = patient_features_folders[:10]
    
    # Determine subfolders where each patient csv file will be written
    num_patients = len(patient_features_folders)
    num_patients_per_subfolder = 1000
    patient_subfolder_ids = [i // num_patients_per_subfolder for i in range(num_patients)]
    patient_write_subfolders = [
        os.path.join(patient_csv_dir, f"split_{subfolder_id:03d}")
        for subfolder_id in patient_subfolder_ids
    ]
    
    # Build patient csv files in, parallelizing wrt patient folders
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(
            process_patient_features_folder, 
            patient_features_folders, 
            patient_write_subfolders,
        ), total=len(patient_features_folders), desc="Building patient csv files"))
    
    return results


def process_patient_features_folder(
    patient_features_folder: str,
    patient_write_subfolder: str,
) -> str:
    """ Process a single patient's data folder, extract feature triplets
        and saving processed triplet data to a common csv file
    """
    # Extract feature triplets from each CSV file in the patient folder
    data_tuples = []
    for filename in os.listdir(patient_features_folder):
        if filename.endswith(".csv") and filename != "icds.csv":
            file_path = os.path.join(patient_features_folder, filename)
            data_tuples.extend(extract_feature_triplets(file_path))
            
    # Sort the tuples by time (i.e., the first column)
    data_tuples.sort(key=lambda x: x[0])
    
    # Save to new CSV file in the processed data directory
    os.makedirs(patient_write_subfolder, exist_ok=True)
    patient_key = os.path.split(patient_features_folder)[-1]
    output_path = os.path.join(patient_write_subfolder, f"{patient_key}.csv")
    df = pd.DataFrame(data_tuples, columns=["times", "types", "values"])
    df.to_csv(output_path, index=False)
    
    
def extract_feature_triplets(
    file_path: str,
) -> list[tuple[float, float, str]]:
    """ Extract a list of added time-feature_type-value triplet for a csv file
        If the csv file contains several features, triplet list will be mixed
    """
    # Load data from feature csv file
    data = pd.read_csv(file_path)
    columns = data.columns
    
    # Dynamic data feature(s)
    subset_time_keys = cfg.TIME_VARS_MAP_DATASET_LIST_KEYS[cfg.RAW_DATA_SUBSET]
    if any(key in columns for key in subset_time_keys):
        if "_ws_" not in file_path and "_ts_" not in file_path:
            warnings.warn("File with dynamic data name: no '_ts_' or '_ws_'! %s" % file_path)
        melted_df = data.melt(id_vars=[columns[0]], var_name="feature_type", value_name="value")
        triplet_vector_list = list(melted_df.itertuples(index=False, name=None))
        
    # Static data feature(s)
    else:
        if "_static_" not in file_path:
            warnings.warn("File with static data name: no '_static_'! %s" % file_path)
        triplet_vector_list = [(0, data[col].iloc[0], col) for col in columns]
        
    return triplet_vector_list


if __name__ == "__main__":
    main()
    