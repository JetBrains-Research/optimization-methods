from torch.nn import Module
from transformers import RobertaForCausalLM, RobertaConfig, RobertaModel, OpenAIGPTConfig, OpenAIGPTLMHeadModel
import torch.nn.functional as F


class CodeBERTa(Module):
    def __init__(self):
        super(CodeBERTa, self).__init__()

        self.encoder = RobertaModel.from_pretrained("huggingface/CodeBERTa-small-v1")
        decoder_config = OpenAIGPTConfig(vocab_size=52000)
        self.decoder = OpenAIGPTLMHeadModel(decoder_config)

        print('ENC', '%.1E' % self.encoder.num_parameters())
        print('DEC', '%.1E' % self.decoder.num_parameters())
        print('total', '%.1E' % (self.encoder.num_parameters() + self.decoder.num_parameters()))

    def forward(self, x):
        embed = self.encoder(x).last_hidden_state
        out = self.decoder(inputs_embeds=embed)
        return out
    
    @staticmethod
    def loss_fn(outputs, targets, batch):
        outputs = outputs.permute(0, 2, 1)
        if outputs.shape[0] != targets.shape[0] or outputs.shape[0] * targets.shape[0] == 0:
            return None
        
        loss = F.cross_entropy(outputs, targets, reduction="none")
        mask = targets != 1
        loss = loss * mask
        return loss.sum() / batch
