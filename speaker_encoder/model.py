import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

TINY = 1e-10
# [1]: GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION (https://arxiv.org/pdf/1710.10467.pdf)

# TODO: verify it works as expected.
class WeightClipper(object):

    def __init__(self, clamp_min=TINY, clamp_max=None):
        self.min = clamp_min
        self.max = clamp_max

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
        else:
            w = module

        if self.max is None:
            w = w.clamp(self.min)
        else:
            w = w.clamp(self.min, self.max)


class SpeakerEncoder(nn.Module):
    def __init__(self, hparams):
        super(SpeakerEncoder, self).__init__()
        self.hidden_size = hparams.lstm_cell_size
        self.depth = hparams.lstm_depth
        self.input_size = hparams.rnn_input_size
        self.dropout = hparams.dropout
        self.pad_token = hparams.pad_token
        self.N = hparams.N      # Nubmer of individuals per batch
        self.M = hparams.M      # Number of audio samples per individual per batch

        self.rnn_network = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.depth,
                                   bias=True,           # Default
                                   batch_first=True,    # Input and output sensors are (batch, seq, feature)
                                   dropout=self.dropout,           # Default
                                   bidirectional=False)

        self.w_similarity = torch.tensor(0.5, dtype=torch.float, requires_grad=True)
        self.b_similarity = torch.tensor(0.5, dtype=torch.float, requires_grad=True)

    def rnn_forward(self, sequences_embedded, len_sequences):
        '''
        Teacher forcing forward path through RNN for training
        '''

        # Order the sequences in descending order or lengths
        # (required for packing the sequence in torch)
        seq_sort_order = np.argsort(len_sequences)[::-1]  # Reverse sort order
        reverse_sort_permutation = np.argsort(seq_sort_order)

        len_sequences = len_sequences[seq_sort_order]
        pack_sequences = nn.utils.rnn.pack_padded_sequence(sequences_embedded,
                                                           lengths=len_sequences,
                                                           batch_first=True)

        # LSTM Inputs: input, (h_0, c_0)
        # If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.
        # LSTM Outputs: output, (h_n, c_n)
        # **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`
        # **c_n** (num_layers * num_directions, batch, hidden_size)
        # **output** of shape `(seq_len, batch, num_directions * hidden_size)` (only last(top) layer)
        rnn_out, (h_n, c_n) = self.rnn_network(pack_sequences)
        c_n = c_n[reverse_sort_permutation]     # Back to the original batch order

        return c_n

    def forward(self, sequences, len_sequences):
        '''
        Training forward path - teacher forcing.
        :param sequences: tensor of label sequences [batch_size]x[max_sequence_len]
                          sorted by length.
        :param len_sequences: np.array, length of sequences, in sorted order.
        :returns: output logits
        '''
        batch_size = sequences.shape[0]
        max_seq_len = sequences.shape[1]

        # ###         RNN Step        # ###
        E = self.rnn_forward(sequences, len_sequences)     # [batch_size] x [embedding_size]
        E = F.normalize(E, p=2, dim=1)      # TODO: Check equation v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

        # Compute the centroid embedding for each of the speakers.
        # [1] Eq. 1
        C = []
        # [1] Eq. 8
        C_minus_i = []

        j = 0
        for i in range(0, self.N * self.M):
            if j == i // self.M:
                c_j = torch.mean(E[j: j + self.M], dim=0, keepdim=True)     # [1]x[embedding_size]
                C.append(c_j)
                j += 1

            c_minus_i = (self.M * c_j + E[i]) / (self.M - 1)    # [1]x[embedding_size]
            C_minus_i.append(c_minus_i)

        # len(C): N
        C_minus_i = torch.cat(C_minus_i, dim=0)     # [batch_size]x[emb_size]

        # [1] Eq. 9
        S = []  # Similarity matrix
        for i, c in enumerate(C):
            s = F.cosine_similarity(E, c, dim=1)    # [batch_size]
            s = s[:, None]     # [batch_size]x[1]

            # [1] Eq. 5: S_{ji, k} = w cos(e_ji, c_k) + b
            s = s * self.w_similarity + self.b_similarity
            S.append(s)

        for i in range(self.N):
            start_ix = i * self.M

            S_i_eq_j = F.cosine_similarity(E[start_ix: start_ix + self.M],
                                           C_minus_i[start_ix: start_ix + self.M],
                                           dim=1)[:, None]

            S[i][start_ix: start_ix + self.M] = S_i_eq_j * self.w_similarity + self.b_similarity

        # len(S): M

        # C = torch.cat(C, dim=0)     # [N]x[embedding_size] - TODO: do I really need this one?
        # S = torch.cat(S, dim=1)     # [batch_size]x[N]
        return S

    def train_batch(self, sequences, len_sequences):
        S = self.forward(sequences, len_sequences)
        # TODO: 
        #   - LOSS COMPUTATION
        #   - BACKWARD
        #   - CLIP GRADIENT FOR W_SIMILARITY


