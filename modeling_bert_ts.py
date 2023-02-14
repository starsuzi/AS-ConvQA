import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
#TODO
import torch.nn.functional as F

from transformers.modeling_bert import (
    BertPreTrainedModel,
    BertModel
)

from transformers.file_utils import(
    add_start_docstrings,
    add_code_sample_docstrings,
)

from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

class BertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, class_num):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, class_num)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = x
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

        
class BertForQuestionAnswering(BertPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, class_num=1):
        super().__init__(config)
        self.class_num = class_num # quac: 1 class (answerable),

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = BertClassificationHead(config, class_num)
        
        #TODO
        self.p = 0.1

        self.init_weights()


    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )

    #TODO
    def mc_dropout(self, x, N):
        x_list = F.dropout(x, p=self.p, training=True)
        x_list = self.qa_outputs(x_list).unsqueeze(0)
        for i in range(N - 1):
            x_tmp = F.dropout(x, p=self.p, training=True)
            x_tmp = self.qa_outputs(x_tmp).unsqueeze(0)
            x_list = torch.cat([x_list, x_tmp], dim=0)
        return x_list    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        is_impossible=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        temp_scale=False, #TODO 
        bayesian=False, #TODO 
        label_smoothing=False, #TODO 
        T=1.0,
        mc_drop_mask_num=10
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
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

        sequence_output = outputs[0]
        _sequence_output = sequence_output

        #TODO
        T = nn.Parameter(torch.tensor(T))
        N = mc_drop_mask_num
        if not temp_scale:
            if not bayesian:
                _sequence_output = F.dropout(_sequence_output, p=self.p, training=self.training)
                logits = self.qa_outputs(_sequence_output)
            else:
                logits = self.mc_dropout(_sequence_output, N)
        else:
            if not bayesian:
                with torch.no_grad():
                    _sequence_output = F.dropout(_sequence_output, p=self.p, training=False)
                    logits = self.qa_outputs(_sequence_output)
                logits = logits / F.relu(T)
            else:
                with torch.no_grad():
                    logits = self.mc_dropout(_sequence_output, N)
                logits = logits / F.relu(T)

        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        class_logits = self.classifier(sequence_output)
        

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            if bayesian:
                ignored_index = start_logits.size(2)#512
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                sf_start_logits = torch.log_softmax(start_logits, dim=2)
                mean_sf_start_logits = sf_start_logits.mean(dim=0)

                sf_end_logits = torch.log_softmax(end_logits, dim=2)
                mean_sf_end_logits = sf_end_logits.mean(dim=0)

                if label_smoothing:
                    loss_fct = LabelSmoothingCrossEntropy()
                    start_loss = loss_fct(mean_sf_start_logits, start_positions, 0.05)
                    end_loss = loss_fct(mean_sf_end_logits, end_positions, 0.05)
                else:
                    start_loss = F.nll_loss(mean_sf_start_logits, start_positions)
                    end_loss = F.nll_loss(mean_sf_end_logits, end_positions)

            else:
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                if label_smoothing:
                    loss_fct = LabelSmoothingCrossEntropy()
                    sf_start_logits = F.log_softmax(start_logits, dim=-1)
                    sf_end_logits = F.log_softmax(end_logits, dim=-1)
                    start_loss = loss_fct(sf_start_logits, start_positions, 0.1)
                    end_loss = loss_fct(sf_end_logits, end_positions, 0.1)
                else:
                    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
            
            if self.class_num < 2: # quac
                class_loss_fct = BCEWithLogitsLoss()
                class_loss = class_loss_fct(class_logits.squeeze(), is_impossible.squeeze())
            
            total_loss = (start_loss + end_loss + class_loss) / 3

        if not return_dict:
            output = (start_logits, end_logits, class_logits)
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )