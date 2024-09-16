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
from typing import Union, Optional


class PatientMLMModel(nn.Module):
    def __init__(
        self,
        original_model_id: str,
        num_value_tokens: int,
        num_type_tokens: int,
    ):
        super().__init__()
        # Initialize LLM from an original pre-trained model
        self.llm = AutoModelForMaskedLM.from_pretrained(original_model_id)
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
        self.llm.cls.predictions.decoder = nn.Linear(
            in_features=self.hidden_size, 
            out_features=num_value_tokens,
            bias=True,
        )
        
        # Make all weights contiguous and untie any tied weight
        self.apply(self._reset_weights_fn)  # "apply" is recursive
        
        # # Tie output decoder linear weights to input value embedding weights
        # value_embedding_weights = self.embedding_layer.value_embedding.weight
        # self.llm.cls.predictions.decoder.weight = value_embedding_weights
        
    @staticmethod
    def _reset_weights_fn(module: nn.Module) -> None:
        """ Make any weight or bias parameter contiguous and untie any shared
            weights in a module by cloning the contiguous parameters
        """
        if hasattr(module, "weight") and module.weight is not None:
            module.weight = nn.Parameter(module.weight.contiguous().clone())
        if hasattr(module, "bias") and module.bias is not None:
            module.bias = nn.Parameter(module.bias.contiguous().clone())
        
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
    ) -> Union[MaskedLMOutput, tuple[torch.Tensor, ...]]:
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
        return self.llm(
            input_ids=None,  # inputs_embeds is used instead
            inputs_embeds=patient_embeddings,
            labels=labels,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


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
    