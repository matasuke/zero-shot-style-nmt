from typing import List, Callable

import torch.nn as nn
import torch

from .model import GlobalAttention
from .model import Model
from .beam import Beam
from preprocessor.text_preprocessor import TextPreprocessor


class Translator(object):
    def __init__(
            self,
            model: Model,
            src_preprocessor: TextPreprocessor,
            tgt_preprocessor: TextPreprocessor,
            replace_unk: bool=True,
            beam_width: int=5,
            n_best: int=1,
            max_length: int=50,
    ):
        self.replace_unk = replace_unk
        self.beam_width = beam_width
        self.n_best = n_best
        self.max_length = max_length

        if torch.cuda.is_available():
            self.tt = torch.cuda
        else:
            self.tt = torch

        self.src_preprocessor = src_preprocessor
        self.tgt_preprocessor = tgt_preprocessor

        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()

        self.model = model
        self.model.eval()

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_preprocessor.indice2tokens(pred, stop_eos=True)
        if self.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == TextPreprocessor.UNK_ID:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, src_batch: torch.Tensor, tgt_batch: torch.Tensor, lengths: List[int]):
        batch_size = src_batch.size(1)

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(src_batch, lengths)

        rnnSize = context.size(2)

        encStates = (self.model._fix_enc_hidden(encStates[0]),
                     self.model._fix_enc_hidden(encStates[1]))

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = src_batch.data.eq(TextPreprocessor.PAD_ID).t()

        def applyContextMask(m):
            if isinstance(m, GlobalAttention):
                m.applyMask(padMask)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = context.data.new_zeros(batch_size)

        if tgt_batch is not None:
            decStates = encStates
            decOut = self.model.make_init_decoder_output(context)
            self.model.decoder.apply(applyContextMask)
            initOutput = self.model.make_init_decoder_output(context)
            decOut, decStates, attn = self.model.decoder(
                tgt_batch[:-1], decStates, context, initOutput)
            for dec_t, tgt_t in zip(decOut, tgt_batch[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(TextPreprocessor.PAD_ID), 0)
                goldScores += scores.view(-1)

        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = context.data.repeat(1, self.beam_width, 1)
        decStates = (encStates[0].data.repeat(1, self.beam_width, 1),
                     encStates[1].data.repeat(1, self.beam_width, 1))

        beam = [Beam(self.beam_width) for k in range(batch_size)]

        decOut = self.model.make_init_decoder_output(context)

        padMask = src_batch.data.eq(TextPreprocessor.PAD_ID).t().unsqueeze(0).repeat(self.beam_width, 1, 1)
        padMask = src_batch.data.eq(TextPreprocessor.PAD_ID).t().repeat(self.beam_width, 1)
        batchIdx = list(range(batch_size))
        remainingSents = batch_size

        for i in range(self.max_length):

            self.model.decoder.apply(applyContextMask)

            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).t().contiguous().view(1, -1)

            decOut, decStates, attn = self.model.decoder(input, decStates, context, decOut)
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            out = self.model.generator.forward(decOut)

            # batch x beam x numWords
            wordLk = out.view(self.beam_width, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(self.beam_width, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue
                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(
                        -1, self.beam_width, remainingSents, decState.size(2))[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return view.index_select(1, activeIdx).view(*newSize)

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)

            padMask = padMask.unsqueeze(2).view(self.beam_width, -1, padMask.size(1))
            padMask = padMask.index_select(1, activeIdx)
            padMask = padMask.view(self.beam_width*activeIdx.size(0), -1)

            remainingSents = len(active)

        #  (4) package everything up

        allHyp, allScores, allAttn = [], [], []
        n_best = self.n_best

        for b in range(batch_size):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best].tolist()]
            valid_attn = src_batch.data[:, b].ne(TextPreprocessor.PAD_ID).nonzero().squeeze(1)
            # hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [[int(i) for i in hyps[0]]]
            allAttn.append([(i.tolist(),) for i in attn])

        return allHyp, allScores, allAttn, goldScores

    def translate(
            self,
            src_batch: torch.Tensor,
            tgt_batch: torch.Tensor,
            lengths: List[int],
    ):
        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(src_batch, tgt_batch, lengths)
        #pred, predScore, attn, goldScore = \
        #    list(zip(*sorted(zip(pred, predScore, attn, goldScore, indices), key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        '''
        predBatch = []
        for b in range(src_batch.size(1)):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], src_batch[b], attn[b][n])
                 for n in range(self.n_best)]
            )
       '''

        return predBatch, predScore, goldScore
