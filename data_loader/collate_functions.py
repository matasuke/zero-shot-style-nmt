from typing import List, Tuple

import torch


def seq2seq_collate_fn(src_tgt_pair: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    '''
    create mini-batch tensors from source target sentences (source_sentence, target_sentence)
    use this collate_fn to pad sentences.

    :param src_tgt_pair: mini batch of source and target sentences
    '''
    def merge(sentences: List[torch.Tensor]):
        '''
        pad sequences for source
        '''
        lengths = [len(sen) for sen in sentences]
        padded_seqs = torch.zeros(len(sentences), max(lengths)).long()

        for idx, sen in enumerate(sentences):
            end = lengths[idx]
            padded_seqs[idx, :end] = sen[:end]

        padded_seqs = padded_seqs.t().contiguous()

        return padded_seqs, lengths

    # sort a list of sentence length based on source sentence to use pad_padded_sequence
    src_tgt_pair.sort(key=lambda x: len(x[0]), reverse=True)
    src, tgt = zip(*src_tgt_pair)

    src, lengths = merge(src)
    tgt, _ = merge(tgt)

    return src, tgt, lengths
