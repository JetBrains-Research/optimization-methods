from torch.nn import Module
from transformers import RobertaModel, OpenAIGPTConfig, OpenAIGPTLMHeadModel, RobertaConfig
import torch.nn.functional as F


class CodeBERTa(Module):
    def __init__(self, roberta_pretrained=False):
        super(CodeBERTa, self).__init__()

        if roberta_pretrained:
            self.encoder = RobertaModel.from_pretrained("huggingface/CodeBERTa-small-v1")
            decoder_config = OpenAIGPTConfig(vocab_size=52000)
            self.decoder = OpenAIGPTLMHeadModel(decoder_config)
        else:
            encoder_config = RobertaConfig(
                vocab_size=52000,
                hidden_size=192, num_hidden_layers=5, num_attention_heads=3, 
                intermediate_size=768, max_position_embeddings=256)
            self.encoder = RobertaModel(encoder_config)
            decoder_config = OpenAIGPTConfig(
                vocab_size=52000, n_ctx=80, n_positions=80,
                n_embd=192)
            self.decoder = OpenAIGPTLMHeadModel(decoder_config)

        print('ENC', '%.1E' % self.encoder.num_parameters())
        print('DEC', '%.1E' % self.decoder.num_parameters())
        print('total', '%.1E' %
              (self.encoder.num_parameters() + self.decoder.num_parameters()))

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
