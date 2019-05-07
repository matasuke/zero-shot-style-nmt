from typing import Tuple, Optional
import torch
import torch.nn as nn


GENERAL_METHOD = 'general'
DOT_METHOD = 'dot'
ATTENTION_METHODS = [GENERAL_METHOD, DOT_METHOD]


class Attention(nn.Module):
    '''
    Apply attention mechanism

    Usage:
        >>> batch_size = 5
        >>> seq_len = 7
        >>> hidden_dim = 256
        >>> attention = Attention(256)
        >>> query = torch.Tensor(batch_size, hidden_dim)
        >>> context = torch.Tensor(batch_size, seq_len, hidden_dim)
        >>> output, attn = attention(query, context)
        >>> output.size()
        torch.Size([5, 1, 256])
        >>> attn.size()
        torch.Size([5, 1, 5])
    '''
    __slots__ = [
        'hidden_dim',
        'attention_type',
        'attn_in',
        'attn_out',
        'softmax',
        'tanh',
        'mask',
    ]

    def __init__(self, hidden_dim: int, attention_type: str=GENERAL_METHOD):
        super(Attention, self).__init__()

        if attention_type not in ATTENTION_METHODS:
            raise ValueError(f'Unknown attention type: {attention_type}')

        self.hidden_dim = hidden_dim
        self.attention_type = attention_type

        if self.attention_type == GENERAL_METHOD:
            self.attn_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_out = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask: Optional[int] = None

    def applyMask(self, mask: torch.Tensor):
        self.mask = mask

    def forward(
            self,
            query: torch.Tensor,
            context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        :param query: Sequence of queries to query the context.
            torch.FloatTensor([batch_size, hidden_dim])
        :param context: Data overwhich to apply the attention mechanism.
            torch.FloatTensor([batch_size, source_max_length, hidden_dim])
        '''
        batch_size, hidden_dim = query.size()
        source_length = context.size(1)

        if self.attention_type == GENERAL_METHOD:
            queryT = self.attn_in(query)
        queryT = queryT.unsqueeze(2)  # [batch_size, hidden_dim, 1]

        # [batch_size, hidden_dim, 1] * [batch_size, source_length, hidden_dim] -> [batch_size, source_length]
        attn_scores = torch.bmm(context, queryT).squeeze(2)

        print('attn', attn_scores.shape)
        print('mask', self.mask.shape)
        if self.mask is not None:
            attn_scores.masked_fill_(self.mask, -float('inf'))

        attn_scores = self.softmax(attn_scores)
        attn_weights = attn_scores.view(batch_size, 1, source_length)  # [batch_size, 1, source_length]

        # [batch_size, 1, source_length] * [batch_size, source_length, hidden_dim] -> [batch_size, hidden_dim]
        weighted_context = torch.bmm(attn_weights, context).squeeze(1)

        # concat([batch_size, hidden_dim], [batch_size, hidden_dim], dim=1) -> [batch_size, hidden_dim*2]
        combined = torch.cat((weighted_context, query), dim=1)

        # [batch_size, hidden_dim*2] -> [batch_size, hidden_dim]
        context_output = self.tanh(self.attn_out(combined))

        return context_output, attn_scores
