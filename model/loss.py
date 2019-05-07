import torch
import torch.nn.functional as F
import torch.nn as nn

from preprocessor.text_preprocessor import TextPreprocessor


def nll_loss(output, target):
    weight = torch.ones(output.size(-1))
    weight[TextPreprocessor.PAD_ID] = 0

    if torch.cuda.is_available():
        weight = weight.cuda()

    return F.nll_loss(output, target, weight)
