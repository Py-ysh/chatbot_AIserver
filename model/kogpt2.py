# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

class KoGPT2Replier(nn.Module):
    def __init__(self):
        super().__init__()
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def reply(self, msg, sent='0'):
        tok = self.tokenizer
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            q = msg
            a = ''
            while True:
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids)
                gen = tok.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS or gen == PAD:
                    break
                a += gen.replace('▁', ' ')
            return f"{a.strip()}"