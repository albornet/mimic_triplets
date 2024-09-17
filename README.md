Here’s a suggested structure for your README file for the repository:

---

# Mimic Triplets: Transformer-based Triplet Embedding for Time-Series Vital Signs

This repository provides a transformer-based model for triplet embeddings on time-series vital signs measurements from the MIMIC dataset. Each measurement is represented as a triplet consisting of the time of measurement, the value, and the type of measurement. These are individually embedded and combined using a BERT-like architecture trained for Masked Language Modeling (MLM) or Causal Language Modeling (Causal LM).

## Installation (TODO: WRITE REQUIREMENT FILES)

```bash
git clone https://github.com/albornet/mimic_triplets.git
cd mimic_triplets
pip install -r requirements.txt
```

## Dataset

The project uses the [MIMIC-III dataset](https://mimic.physionet.org/) for training. To access the data, you must request access from the official [MIMIC-III page](https://mimic.physionet.org/gettingstarted/access/).

## Preprocessing (NOT THERE YET)

Vital signs from MIMIC-III are processed into triplets `(time, value, measurement_type)` before being fed into the model. Use the following command to preprocess:

```bash
python preprocess_data.py --input_path <input_path> --output_path <output_path>
```

## Training (TODO: TURN CONFIG.PY INTO CONFIG.YAML)

To train the model, use the following command:

```bash
python train.py --config config.yaml
```

Training options such as batch size, learning rate, and model architecture can be modified in `config.yaml`.

## Evaluation (NOT THERE YET)

Evaluation is performed by calculating prediction accuracy on masked tokens in the MLM framework. To run evaluation:

```bash
python evaluate.py --model_path <model_path> --data_path <data_path>
```