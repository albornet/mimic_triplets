import random
import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import (
    MaskedLMOutput,
    CausalLMOutput,
)
from transformers.data.data_collator import DataCollatorMixin
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Union, Optional


class PatientMLMModel(nn.Module):
    def __init__(
        self,
        original_model_id: str,
        language_model_type: str,
        num_value_tokens: int,
        num_type_tokens: int,
    ):
        super().__init__()
        assert language_model_type in ["masked", "causal"],\
            "language_model_type must be masked or causal"
        
        # Initialize LLM from an original pre-trained model
        if language_model_type == "masked":
            self.llm = AutoModelForMaskedLM.from_pretrained(original_model_id)
        elif language_model_type == "causal":
            self.llm = AutoModelForCausalLM.from_pretrained(original_model_id)
        self.hidden_size = self.llm.config.hidden_size
        self.num_tokens_max = self.llm.config.max_position_embeddings
        
        # Create embedding layer and replace the LLM one to prevent incompatibility
        self.embedding_layer = PatientEmbedding(
            embedding_dim=self.hidden_size,
            num_value_tokens=num_value_tokens,
            num_type_tokens=num_type_tokens,
        )
        
        # Modify the MLM head (classifier) to match the number of value tokens
        self.llm.config.vocab_size = num_value_tokens
        new_decoder_layer = nn.Linear(
            in_features=self.hidden_size, 
            out_features=num_value_tokens,
            bias=True,
        )
        if language_model_type == "masked":
            self.llm.cls.predictions.decoder = new_decoder_layer
        elif language_model_type == "causal":
            self.llm.lm_head = new_decoder_layer
        
        # Make all weights contiguous and untie any tied weight
        self.apply(self._reset_weights_fn)  # "apply" is recursive
        
        # # Tie output decoder linear weights to input value embedding weights
        # value_embedding_weights = self.embedding_layer.value_embedding.weight
        # self.llm.cls.predictions.decoder.weight = value_embedding_weights
        
    def _reset_weights_fn(self, module: nn.Module) -> None:
        """ Make any weight or bias parameter contiguous and untie any shared
            weights in a module by cloning the contiguous parameters
        """
        try:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight = nn.Parameter(module.weight.contiguous().clone())
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = nn.Parameter(module.bias.contiguous().clone())
        except RuntimeError:
            if not hasattr(self, "already_printed_warnings"):
                self.already_printed_warnings = set()
            warning = "Module %s was not reset!" % module._get_name()
            if warning not in self.already_printed_warnings:
                print(warning)
            self.already_printed_warnings.add(warning)
            
    def forward(
        self,
        times: torch.Tensor,
        values: torch.Tensor,
        types: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        head_mask: Optional[torch.Tensor]=None,
        labels: Optional[torch.LongTensor]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
    ) -> Union[MaskedLMOutput, CausalLMOutput, tuple[torch.Tensor, ...]]:
        """ Masked LM forward function adapted for patient embeddings
        
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss
            Indices should be in [-100, 0, ..., config.vocab_size]
            Tokens with indices set to -100 are ignored (masked), the loss is
            only computed for the tokens with labels in [0, ..., config.vocab_size].
        """
        # Generate patient embeddings
        patient_embeddings = self.embedding_layer(times, values, types)
        
        # Forward to the LLM model using inputs_embeds
        output = self.llm(
            input_ids=None,  # inputs_embeds is used instead
            inputs_embeds=patient_embeddings,
            labels=labels,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return output


class PatientEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_value_tokens: int,
        num_type_tokens: int,
    ):
        """_summary_

        Args:
            embedding_dim (int): output dimension of the generated embeddings
            type_vocab (dict[str, int]): vocabulary for the "type" tokens
        """
        super().__init__()
        
        self.time_linear = nn.Linear(1, embedding_dim)
        self.value_embedding = nn.Embedding(num_value_tokens, embedding_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(num_type_tokens, embedding_dim, padding_idx=0)
        
        self.layer_norm = nn.LayerNorm((embedding_dim,), eps=1e-05, elementwise_affine=True)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        
    def forward(
        self,
        times: torch.Tensor,
        values: torch.Tensor,
        types: torch.Tensor,
    ):
        """ Generate a unified embedding tensor for patient measurement events
            that come with time, value, and type of measurements
        
        Args:
            times (torch.Tensor): time of measurements
            values (torch.Tensor): value of measurements
            types (torch.Tensor): type of measurements

        Returns:
            torch.Tensor: output embedding tensor for patient measurements
        """
        # Embed each input tensor
        time_embed = self.time_linear(times)
        value_embed = self.value_embedding(values)
        type_embed = self.type_embedding(types)
        
        # Combine embeddings into a single tensor
        combined_embeddings = time_embed + value_embed + type_embed
        combined_embeddings = self.layer_norm(combined_embeddings)
        combined_embeddings = self.dropout(combined_embeddings)
        
        return combined_embeddings


@dataclass
class PatientDataCollatorForLanguageModelling(DataCollatorMixin):
    """ Data collator used for the PatientEmbedding-based language model
        Modified from transformers.data.data_collator.py
    
    Args:
        mlm (bool): whether or not to use masked language modeling
        mlm_probability(float): probability with which tokens are mask randomly
    """
    mlm: bool=True
    mask_id: int=1
    pad_id: int=0
    bos_id: int=2
    eos_id: int=3
    num_tokens_max: int=512
    num_mlm_labels: Optional[int]=None
    mlm_probability: float=0.15
    return_tensors: str="pt"
    
    def torch_call(
        self,
        samples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """ Collate patient embedding samples and create mlm labels for training
        """
        # Control each sample for sequence length and add bos / eos token ids
        effective_num_token_max = self.num_tokens_max - 2   # for bos and eos tokens
        for batch_idx, _ in enumerate(samples):
            seq_len = next(iter(samples[batch_idx].values())).shape[0]
            if seq_len > effective_num_token_max:
                
                # Randomly select a starting index for slicing
                start_idx = random.randint(0, seq_len - effective_num_token_max)
                end_idx = start_idx + effective_num_token_max
                
                # Slice all tensors in the sample by keys (with the same slicing)
                for data_key in samples[batch_idx]:
                    random_slice = samples[batch_idx][data_key][start_idx:end_idx]
                    samples[batch_idx][data_key] = random_slice
        
        # Add token ids for beginning and end of sequence
        for batch_idx, _ in enumerate(samples):
            for data_key in samples[batch_idx]:
                to_enclose = samples[batch_idx][data_key]
                enclosed = self.add_bos_eos_ids(to_enclose, data_key)
                samples[batch_idx][data_key] = enclosed
                
        # Create batch object by padding time, value, and type sequences
        times = [e["times"] for e in samples]
        values = [e["values"] for e in samples]
        types = [e["types"] for e in samples]
        batch = {
            "times": pad_sequence(times, batch_first=True, padding_value=0.0),
            "values": pad_sequence(values, batch_first=True, padding_value=0),
            "types": pad_sequence(types, batch_first=True, padding_value=0),
        }
        
        # Mask values and record original values as labels if MLM is used
        if self.mlm:
            assert self.num_mlm_labels is not None, "Define the number of labels for mlm"
            batch["values"], batch["labels"] = self.masked_modelling(batch["values"])
        
        # If MLM is not used, causal language modelling labels are generated 
        else:
            batch["labels"] = self.causal_modelling(batch["values"])
        
        return batch
        
    def add_bos_eos_ids(
        self,
        sequence: torch.Tensor,
        data_key: str,
    ) -> torch.Tensor:
        """ Add bos and eos token ids or first and last time to a sequence
            given the data it contains
        """
        if data_key == "times":
            to_add = [sequence[0].unsqueeze(0), sequence[-1].unsqueeze(0)]
        else:
            to_add = [torch.tensor([self.bos_id]), torch.tensor([self.eos_id])]
        
        return torch.cat([to_add[0], sequence, to_add[-1]], dim=0)
    
    def masked_modelling(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs and labels for masked language modeling
            Modified from transformers.data.data_collator.py
        """
        # Prepare labels and mask array
        labels = inputs.clone()  # labels are the unmasked version of inputs
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Sample tokens in each sequence for mlm training, using mlm_probability
        probability_matrix.masked_fill_(labels == self.pad_id, value=0.0)  # ignore pad tokens
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # special code to only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with mask token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_id
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.num_mlm_labels, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels
    
    def causal_modelling(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """ Prepare labels for causal language modeling by shifting tokens to the right
            Modified from transformers.data.data_collator.py
        """
        labels = inputs.clone()
        if self.pad_id is not None:
            labels[labels == self.pad_id] = -100
        
        return labels