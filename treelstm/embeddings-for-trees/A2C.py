import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from models import TreeLSTM2Seq
import dgl
from omegaconf import DictConfig
from data_module.jsonl_data_module import JsonlDataModule
from utils.training import cut_encoded_data
from utils.common import EOS
# from nltk.translate.bleu_score import sentence_bleu
from torchtext.data.metrics import bleu_score
import numpy as np


class Actor(nn.Module):
    def __init__(self, config, vocab, path_to_pretrained_actor):
        super().__init__()
        model = TreeLSTM2Seq(config, vocab)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path_to_pretrained_actor, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        self.embedding = model._embedding
        self.encoder = model._encoder
        self.decoder = model._decoder
        self._cur_encoded_trees = None
        self._cur_attention_mask = None
        self._cur_pred_seq = None

    def encode(  # type: ignore
            self,
            batched_trees: dgl.DGLGraph,
    ) -> torch.Tensor:
        batched_trees.ndata["x"] = self.embedding(batched_trees)
        encoded_nodes = self.encoder(batched_trees)
        return encoded_nodes, batched_trees.batch_num_nodes()

    def policy(self, cur_input, h_prev, c_prev, batched_encoded_trees, attention_mask):
        current_output, (h_prev, c_prev) = self.decoder.decoder_step(
                cur_input, h_prev, c_prev, batched_encoded_trees, attention_mask
            )

        distr = F.softmax(current_output, dim=-1)

        return distr, h_prev, c_prev


class Critic(nn.Module):
    def __init__(self, state_dim=320):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, tree_term, hidden_term):
        inp = tree_term.mean(dim=1) + hidden_term.squeeze(0)
        return self.model(inp).squeeze()


EOS_IDX = 3


def pretrain_critic(actor, state_dim, train_loader, train_params):
    critic = Critic(state_dim)
    # for _ in range(train_params['epochs']):
    #     for states, labels in iter(train_loader):
    #         with torch.no_grad():
    #             pred = actor(states)
    #         #TODO

    return critic


def a2c(path_to_pretrained, config, train_params):
    torch.manual_seed(train_params['seed'])
    data_module = JsonlDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    actor = Actor(config, data_module.vocabulary, path_to_pretrained)
    critic = pretrain_critic(
        actor,
        train_params['state_dim'],
        data_module.train_dataloader(),
        train_params['critic_pretraining'],
    )

    actor_optimizer = Adam(actor.parameters(), lr=train_params['actor_lr'])
    critic_optimizer = Adam(critic.parameters(), lr=train_params['critic_lr'])
    all_rewards = []
    entropy_term = 0

    for epoch in range(train_params['epochs']):
        for step, (labels, batched_trees) in enumerate(data_module.train_dataloader()):
            log_probs, values, rewards = [], [], []
            encoded_trees, tree_sizes = actor.encode(batched_trees)
            batched_encoded_trees, attention_mask = cut_encoded_data(encoded_trees, tree_sizes,
                                                                                 actor.decoder._negative_value)
            initial_state = (
                torch.cat([ctx_batch.mean(0).unsqueeze(0) for ctx_batch in encoded_trees.split(tree_sizes.tolist())])
                    .unsqueeze(0)
                    .repeat(actor.decoder._decoder_num_layers, 1, 1)
            )
            h_prev, c_prev = initial_state, initial_state
            batch_size = tree_sizes.shape[0]
            cur_input = encoded_trees.new_full((batch_size,), actor.decoder._sos_token, dtype=torch.long)

            output_length = train_params['output_length']
            output = encoded_trees.new_zeros((output_length, batch_size), dtype=torch.long)
            dones = torch.zeros(batch_size, dtype=torch.long)
            done_mask = []

            for t in range(output_length):
                policy_distr, h_curr, c_curr = actor.policy(
                    cur_input, h_prev, c_prev, batched_encoded_trees, attention_mask
                )

                distr = torch.distributions.Categorical(probs=policy_distr)
                actions = distr.sample()
                output[t] = actions.detach()
                cur_input = actions
                dones[dones == 1] = 2
                dones[(dones == 0) & (actions == EOS_IDX)] = 1

                entropy_term += distr.entropy().sum()

                # for batch_idx in range(batch_size):
                #     cur_distr = distr[batch_idx]
                #     action = np.random.choice(actor.decoder._out_size, p=cur_distr)
                #     if dones[batch_idx] == 0 and action == EOS_IDX:
                #         dones[batch_idx] = 1  # first eos
                #     elif dones[batch_idx] == 1:
                #         dones[batch_idx] = 2  # after eos
                #     outputs_batch[batch_idx] = action
                #     log_probs_batch[batch_idx] = torch.log(policy_distr[batch_idx, action])
                #     entropy = -np.sum(np.mean(cur_distr) * np.log(cur_distr))
                #     entropy_term += entropy

                done_mask.append(dones.clone())
                value_batch = critic(batched_encoded_trees, h_prev)

                # output[t] = torch.tensor(outputs_batch, dtype=torch.long)
                reward_batch = compute_rewards(output, labels, t, output_length, dones)
                rewards.append(reward_batch)
                values.append(value_batch)
                log_probs.append(distr.log_prob(actions))

                h_prev, c_prev = h_curr, c_curr

                if all(dones) or t == output_length - 1:
                    Qval = critic(batched_encoded_trees, h_curr)
                    Qval = Qval.detach().numpy()
                    all_rewards.extend(np.sum(rewards, axis=0))
                    if step % 10 == 0:
                        print(
                            f"Epoch: {epoch}, step: {step}, reward mean: {np.mean(np.sum(rewards, axis=0))},"
                            f"rewards std: {np.std(np.sum(rewards, axis=0))}"
                        )
                    break

            # compute target Q value
            done_mask = (torch.stack(done_mask) < 2).type(torch.float).detach()
            values = torch.stack(values)  # critic gradients here
            Qvals = np.zeros((t + 1, batch_size))
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + train_params['GAMMA'] * Qval
                Qvals[t] = Qval

            # values = torch.FloatTensor(values)
            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)  # actor gradients here

            assert values.size() == done_mask.size() == Qvals.size() == log_probs.size(), "Sizes mismatch"

            advantage = (Qvals - values) * done_mask
            actor_loss = (-log_probs * advantage.detach()).mean() # + 0.001 * entropy_term
            actor_optimizer.zero_grad()

            # print('Advantage:', advantage)
            # print('done_mask:', done_mask)
            # print('log_probs:', log_probs)
            #
            # print('ACTOR LOSS:', actor_loss)
            #
            # assert False, 'exit'

            # critic_loss = 0.5 * advantage.pow(2).mean()
            # critic_optimizer.zero_grad()

            #update
            actor_loss.backward()
            # critic_loss.backward()
            actor_optimizer.step()
            # critic_optimizer.step()

    return actor


def bleu_n_gramm_weights(l):
    length = len(l)
    if length >= 4:
        return 4, (0.25, 0.25, 0.25, 0.25)
    else:
        return length, tuple(1 / length for _ in range(length))


def compute_rewards(pred_seq, target, time_stamp, out_len, dones, sparse=False):
    batch_size = len(dones)
    res = [0] * batch_size
    for batch_idx in range(batch_size):
        cur_tgt = target[:, batch_idx].tolist()
        cur_tgt = [[list(map(str, cur_tgt[:cur_tgt.index(EOS_IDX)+1]))]]
        cur_hyp = list(map(str, pred_seq[:time_stamp+1, batch_idx].tolist()))
        cur_max_n, cur_weights = bleu_n_gramm_weights(cur_hyp)
        prev_hyp = list(map(str, pred_seq[:time_stamp, batch_idx].tolist()))
        prev_max_n, prev_weights = bleu_n_gramm_weights(prev_hyp)
        if not sparse:
            if dones[batch_idx] < 2:
                cur_bleu = bleu_score([cur_hyp], cur_tgt, max_n=cur_max_n, weights=cur_weights)
                prev_bleu = bleu_score([prev_hyp], cur_tgt, max_n=prev_max_n, weights=prev_weights) if prev_max_n else 0
                res[batch_idx] = cur_bleu - prev_bleu
            else:
                res[batch_idx] = 0
            # print('CUR TGT:', cur_tgt, 'CUR HYP:', cur_hyp, 'reward:', res[batch_idx], 'done:', dones[batch_idx])
        else:
            if dones[batch_idx] == 1 or time_stamp == out_len:
                res[batch_idx] = bleu_score([cur_hyp], cur_tgt, max_n=cur_max_n, weights=cur_weights)
            else:
                res[batch_idx] = 0
    return res


if __name__ == '__main__':
    train_params = {
        'epochs': 1,
        'lr': 1e-3,
    }

