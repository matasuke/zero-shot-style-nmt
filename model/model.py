from typing import Optional, Union, Tuple, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from preprocessor.text_preprocessor import TextPreprocessor
from .global_attention import GlobalAttention


class Encoder(nn.Module):
    __slots__ = [
        'emb_dim',
        'vocab_size',
        'num_layers',
        'dropout_ratio',
        'num_directions',
        'hidden_dim',
        'embedding',
        'lstm',
    ]

    def __init__(
            self,
            emb_dim: int,
            vocab_size: int,
            hidden_dim: int,
            num_layers: int,
            brnn: bool=True,
            dropout_ratio: float=0,
    ):
        '''
        seq2seq encoder

        :param emb_dim: dimention of embedding layer
        :param vocab_size: vocabulary size
        :param hidden_dim: hidden dimention
        :param num_layers: the number of layers
        :param brnn: use bidirectional LSTM
        :param dropout_ratio: dropout ratio
        '''
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.num_directions = 2 if brnn else 1
        assert hidden_dim % self.num_directions == 0
        self.hidden_dim = hidden_dim // self.num_directions

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.emb_dim,
            padding_idx=TextPreprocessor.PAD_ID,
        )
        self.lstm = nn.LSTM(
            self.emb_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_ratio,
            bidirectional=brnn,
        )

    def load_embedding(self, emb_path: Union[str, Path]):
        '''
        Load pretrained embedding.
        '''
        if isinstance(emb_path, str):
            emb_path = Path(emb_path)
        assert emb_path.exists()

        pretrained = torch.load(emb_path.as_posix())
        self.embedding.weight.data.copy_(pretrained)

    def init_hidden_state(self):
        return torch.zeros(2, self.hidden_size)

    def forward(
            self,
            rnn_inputs: torch.Tensor,
            lengths: List[int],
            hidden_states: Optional[torch.Tensor]=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        forward pass for Encoder

        :param rnn_inputs: tensors of padded tokens size of [batch_size, max_len]
        :param hidden_state: hidden state to initialize LSTM state.
        '''
        embedded = self.embedding(rnn_inputs)
        embedded = pack_padded_sequence(embedded, lengths)
        outputs, hidden_states = self.lstm(embedded, hidden_states)
        outputs, _ = pad_packed_sequence(outputs)

        return hidden_states, outputs


class StackedLSTM(nn.Module):

    __slots__ = [
        'dropout',
        'num_layers',
        'layers'
    ]

    def __init__(
            self,
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            dropout_ratio: float=0.3,
    ):
        '''
        StackedLSTM to feed data just one step, which is useful for attention mechanism.

        :param num_layers: the number of layers
        :param input_dim: dimension of input
        :param hidden_dim: dimenstion of hidden layers
        :param dropout_ratio: dropout ratio
        '''
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout_ratio)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_dim, hidden_dim))
            input_dim = hidden_dim

    def forward(
            self,
            rnn_inputs: torch.Tensor,
            hidden_states: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_0, c_0 = hidden_states
        h_1: List[torch.Tensor] = []
        c_1: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(rnn_inputs, (h_0[i], c_0[i]))
            rnn_inputs = h_1_i
            if i + 1 != self.num_layers:
                rnn_inputs = self.dropout(rnn_inputs)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return rnn_inputs, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(
            self,
            emb_dim: int,
            vocab_size: int,
            hidden_dim: int,
            num_layers: int,
            input_feed: bool=False,
            dropout_ratio: float=0,
    ):
        '''
        seq2seq decoder
        '''
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.input_feed = input_feed

        if self.input_feed:
            self.input_dim = emb_dim + hidden_dim
        else:
            self.input_dim = hidden_dim

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=TextPreprocessor.PAD_ID,
        )
        self.lstm = StackedLSTM(num_layers, self.input_dim, hidden_dim, dropout_ratio)
        self.attn = GlobalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def load_embedding(self, emb_path: Union[str, Path]):
        '''
        Load pretrained embedding.
        '''
        if isinstance(emb_path, str):
            emb_path = Path(emb_path)
        assert emb_path.exists()

        pretrained = torch.load(emb_path.as_posix())
        self.embedding.weight.data.copy_(pretrained)

    def forward(
            self,
            rnn_inputs: torch.Tensor,
            hidden_state: torch.Tensor,
            context: torch.Tensor,
            init_output: torch.Tensor,
    ) -> torch.Tensor:
        '''
        :param rnn_inputs:
        :param hidden_state:
        '''
        embedded = self.embedding(rnn_inputs)

        outputs: torch.Tensor = []
        output = init_output
        for emb_t in embedded.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden_state = self.lstm(emb_t, hidden_state)
            output, attn = self.attn(output, context.transpose(0, 1))
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden_state, attn


class Model(nn.Module):
    '''
    Model for attentional seq2seq
    '''
    def __init__(
            self,
            emb_dim: int,
            in_vocab_size: int,
            out_vocab_size: int,
            hidden_dim: int,
            num_layers: int=3,
            dropout_ratio: float=0.3,
            brnn: bool=True,
            input_feed: bool=False,
    ):
        super(Model, self).__init__()
        self.encoder = Encoder(
            emb_dim,
            in_vocab_size,
            hidden_dim,
            num_layers,
            brnn,
            dropout_ratio,
        )
        self.decoder = Decoder(
            emb_dim,
            out_vocab_size,
            hidden_dim,
            num_layers,
            input_feed,
            dropout_ratio,
        )
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, out_vocab_size),
            nn.LogSoftmax(dim=1),
        )

    def make_init_decoder_output(self, context: torch.Tensor):
        batch_size = context.size(1)
        return context.new_zeros(batch_size, self.decoder.hidden_dim)

    def _fix_enc_hidden(self, hidden_state: torch.Tensor):
        # [num_layers*directions, batch_size, hidden_dim] -> [num_layers, batch_size, directions*hidden_dim]
        if self.encoder.num_directions == 2:
            return hidden_state.view(hidden_state.size(0) // 2, 2, hidden_state.size(1), hidden_state.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(hidden_state.size(0) // 2, hidden_state.size(1), hidden_state.size(2) * 2)
        else:
            hidden_state

    def forward(self, rnn_inputs: torch.Tensor, targets: torch.Tensor, lengths: List[int]):
        enc_hidden, context = self.encoder(rnn_inputs, lengths)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]), self._fix_enc_hidden(enc_hidden[1]))
        out, dec_hidden, _attn = self.decoder(targets, enc_hidden, context, init_output)

        out = self.generator(out.view(-1, out.size(2)))

        return out
