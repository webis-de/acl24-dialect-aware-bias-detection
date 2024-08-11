"""
Implementation inspired by transformers package and extended to support multiple prediction heads:

https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
"""

from typing import Optional, Tuple, Union

import torch
import transformers
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION as _BERT_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
)
from transformers.models.bert.modeling_bert import _CONFIG_FOR_DOC as _BERT_CONFIG_FOR_DOC
from transformers.models.bert.modeling_bert import (
    _SEQ_CLASS_EXPECTED_LOSS as _BERT_SEQ_CLASS_EXPECTED_LOSS,
)
from transformers.models.bert.modeling_bert import (
    _SEQ_CLASS_EXPECTED_OUTPUT as _BERT_SEQ_CLASS_EXPECTED_OUTPUT,
)
from transformers.models.bert.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    BertModel,
    BertPreTrainedModel,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    _CHECKPOINT_FOR_DOC as _DEBERTAV2_CHECKPOINT_FOR_DOC,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    _CONFIG_FOR_DOC as _DEBERTAV2_CONFIG_FOR_DOC,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DEBERTA_INPUTS_DOCSTRING,
    ContextPooler,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    StableDropout,
)
from transformers.models.roberta.modeling_roberta import (
    _CONFIG_FOR_DOC,
    ROBERTA_INPUTS_DOCSTRING,
    RobertaModel,
    RobertaPreTrainedModel,
)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = kwargs.get("task_labels_map", {})
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # Add task-specific output heads
        for task, output_dim in self.num_labels.items():
            setattr(self, f"classifier_{task}", nn.Linear(config.hidden_size, output_dim))

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_BERT_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_BERT_CONFIG_FOR_DOC,
        expected_output=_BERT_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_BERT_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_name: str = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = getattr(self, f"classifier_{task_name}")(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels[task_name] == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels[task_name] > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels[task_name] == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels[task_name]), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(transformers.PretrainedConfig())
        self.num_labels = kwargs.get("task_labels_map", {})
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Roberta classificaiton head
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # Add task-specific output heads
        for task, output_dim in self.num_labels.items():
            setattr(self, f"classifier_{task}", nn.Linear(config.hidden_size, output_dim))

        self.post_init()

    @add_start_docstrings_to_model_forward(
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_name: str = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be
            in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss
            is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0][:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = getattr(self, f"classifier_{task_name}")(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels[task_name] == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels[task_name] > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels[task_name] == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels[task_name]), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # self.log_writer.add_scalar(tag=f"train/{task_name}_loss", scalar_value=loss, global_step=)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(transformers.PretrainedConfig())

        self.num_labels = kwargs.get("task_labels_map", {})
        self.config = config

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Add task-specific output heads
        for task, output_dim in self.num_labels.items():
            setattr(self, f"classifier_{task}", nn.Linear(config.hidden_size, output_dim))

        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    @add_start_docstrings_to_model_forward(
        DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_DEBERTAV2_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_DEBERTAV2_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_name: str = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be
            in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss
            is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = getattr(self, f"classifier_{task_name}")(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels[task_name] == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(
                            labeled_logits.view(-1, self.num_labels[task_name]).float(),
                            labels.view(-1),
                        )
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels[task_name] == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels[task_name]), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
