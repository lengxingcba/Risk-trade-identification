import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


class Conv(nn.Module):
    def __init__(self, inp, oup, k, s=1):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=inp, out_channels=oup, kernel_size=k, stride=s, padding=k // 2, bias=False)
        self.BN = nn.BatchNorm1d(oup)
        self.act = nn.Mish()

    def forward(self, x):
        return self.act(self.BN(self.conv1d(x)))


class encoder(nn.Module):
    def __init__(self, d_model, n_head=2, num_layers=2):
        super().__init__()
        encoderlayer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=256)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer=encoderlayer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class Block(nn.Module):
    def __init__(self, inp, oup, k, shortcut=True, nhead=2, nlayers=2, stride=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nhead = nhead
        self.nlayers = nlayers
        self.shortcut = shortcut
        self.inp = inp
        self.oup = oup
        self.conv = Conv(inp, oup, k, s=stride)

    def forward(self, x):
        indentity = x
        x = self.conv(x)
        feature_lenth = x.shape[2]
        transformerlayer = nn.TransformerEncoderLayer(d_model=feature_lenth, nhead=self.nhead)
        transformer_encoder = nn.TransformerEncoder(transformerlayer, self.nlayers).to(device=self.device)

        assert feature_lenth // self.nhead != 0, 'output dim must be devided by nhead'

        return indentity + transformer_encoder(x) if self.inp == self.oup and self.shortcut else transformer_encoder(x)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classses = num_classes

    def forward(self, x):
        b, c, f = x.shape

        decoder = nn.Linear(c * f, self.num_classses).to(self.device)
        x = x.view(b, -1)
        return decoder(x)


class Decoder_softmax(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classses = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, f = x.shape
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return self.softmax(x)


class MLP(nn.Module):
    def __init__(self, inp=1, num_classes=2):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(inp, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, f = x.shape
        x = x.view(b, c * f)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return self.softmax(x)
