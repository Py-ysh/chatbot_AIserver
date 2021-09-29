import torch
import torch.nn as nn
from kobert_transformers import get_tokenizer, get_kobert_model

class KoBERTClassifier(nn.Module):
    def __init__(self, num_labels = 58, hidden_size = 768, hidden_dropout_prob = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.kobert = get_kobert_model()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.tokenizer = get_tokenizer()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        outputs = self.kobert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def classify(self, msg, label_encoder, max_seq_len = 512):
            index_of_words = self.tokenizer.encode(msg)
            token_type_ids = [0] * len(index_of_words)
            attention_mask = [1] * len(index_of_words)

            # Padding Length
            padding_length = max_seq_len - len(index_of_words)

            # Zero Padding
            index_of_words += [0] * padding_length
            token_type_ids += [0] * padding_length
            attention_mask += [0] * padding_length

            data = {
                'input_ids': torch.tensor([index_of_words]),
                'token_type_ids': torch.tensor([token_type_ids]),
                'attention_mask': torch.tensor([attention_mask]),
            }
            
            output = self(**data)

            logit = output[0]
            softmax_logit = torch.softmax(logit,dim=-1)
            softmax_logit = softmax_logit.squeeze()

            max_index = torch.argmax(softmax_logit).item()
            # max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

            category = label_encoder.inverse_transform([max_index])
            return f"{category[0]}"