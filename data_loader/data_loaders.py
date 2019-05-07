from typing import Callable, Union, List
from pathlib import Path

import torch
from torch.utils.data import Dataset

from base import BaseDataLoader
from preprocessor import TextPreprocessor
from .collate_functions import seq2seq_collate_fn


class Seq2SeqDataset(Dataset):
    '''
    Dataset for seq2seq
    '''
    __slots__ = [
        'src_list',
        'tgt_list',
        'src_text_preprocessor',
        'tgt_text_preprocessor',
    ]

    def __init__(
            self,
            src_list: List[str],
            src_text_preprocessor: TextPreprocessor,
            tgt_list: List[str],
            tgt_text_preprocessor: TextPreprocessor,
    ):
        '''
        create seq2seq dataset.

        :param src_list: list of source text
        :param src_text_preprocessor: source text preprocessor
        :param tgt_list: list of target text
        :param tgt_text_preprocessor: target text preprocessor
        '''
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.src_text_preprocessor = src_text_preprocessor
        self.tgt_text_preprocessor = tgt_text_preprocessor
        assert len(src_list) == len(tgt_list)

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        src_tokens = self.src_list[idx].split()
        src_indices = self.src_text_preprocessor.tokens2indice(src_tokens, sos=False, eos=False)
        src_indices = torch.Tensor(src_indices)

        tgt_tokens = self.tgt_list[idx].split()
        tgt_indices = self.tgt_text_preprocessor.tokens2indice(tgt_tokens, sos=True, eos=True)
        tgt_indices = torch.Tensor(tgt_indices)

        return src_indices, tgt_indices

    @classmethod
    def create(
            cls,
            source_path: Union[str, Path],
            source_text_preprocessor: TextPreprocessor,
            target_path: Union[str, Path],
            target_text_preprocessor: TextPreprocessor,
    ) -> 'Seq2SeqDataset':
        '''
        create seq2seq dataset from text paths

        :param source_path: path to source sentences
        :param target_path: path to target sentences
        :param source_text_preprocessor: source text preprocessor
        :param target_text_preprocessor: target text preprocessor
        '''
        if isinstance(source_path, str):
            source_path = Path(source_path)
        if isinstance(target_path, str):
            target_path = Path(target_path)
        assert source_path.exists()
        assert target_path.exists()
        with source_path.open() as f:
            source_text_list = [text.strip().lower() for text in f.readlines()]

        with target_path.open() as f:
            target_text_list = [text.strip().lower() for text in f.readlines()]

        assert len(source_text_list) == len(target_text_list)

        return cls(
            source_text_list,
            source_text_preprocessor,
            target_text_list,
            target_text_preprocessor,
        )


class Seq2seqDataLoader(BaseDataLoader):
    '''
    Seq2Seq data loader using BaseDataLoader
    '''
    def __init__(
            self,
            src_path: Union[str, Path],
            src_preprocessor_path: Union[str, Path],
            tgt_path: Union[str, Path],
            tgt_preprocessor_path: Union[str, Path],
            batch_size: int=1,
            shuffle: bool=True,
            validation_split: float=0.0,
            num_workers: int=1,
            collate_fn: Callable=seq2seq_collate_fn,
    ):
        '''
        DataLoader for seq2seq data

        :param source_path: path to source sentences
        :param target_path: path to target sentences
        :param source_text_preprocessor: source text preprocessor
        :param target_text_preprocessor: target text preprocessor
        :param batch_size: batch size
        :param shuffle: shuffle data
        :param validation_split: split dataset for validation
        :param num_workers: the number of workers
        '''
        if isinstance(src_path, str):
            src_path = Path(src_path)
        if isinstance(tgt_path, str):
            tgt_path = Path(tgt_path)
        if isinstance(src_preprocessor_path, str):
            src_preprocessor_path = Path(src_preprocessor_path)
        if isinstance(tgt_preprocessor_path, str):
            tgt_preprocessor_path = Path(tgt_preprocessor_path)
        assert src_path.exists()
        assert tgt_path.exists()
        assert src_preprocessor_path.exists()
        assert tgt_preprocessor_path.exists()
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.src_preprocessor_path = src_preprocessor_path
        self.tgt_preprocessor_path = tgt_preprocessor_path
        self.src_text_preprocessor = TextPreprocessor.load(src_preprocessor_path)
        self.tgt_text_preprocessor = TextPreprocessor.load(tgt_preprocessor_path)

        self.dataset = Seq2SeqDataset.create(
            src_path,
            self.src_text_preprocessor,
            tgt_path,
            self.tgt_text_preprocessor,
        )

        super(Seq2seqDataLoader, self).__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=collate_fn,
        )
