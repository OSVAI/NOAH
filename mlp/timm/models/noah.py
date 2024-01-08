import torch
import torch.nn as nn


class NOAH(nn.Module):
    def __init__(self, inplanes, outplanes, dropout=0.0, key_ratio=0.5, head_num=1, head_split=True, kv_split=True):
        super(NOAH, self).__init__()
        self.kv_split = kv_split
        self.head_split = head_split
        self.dropout = nn.Dropout(p=dropout)
        self.key_ratio = key_ratio
        self.head_num = head_num

        if kv_split:
            self.k_channel = int(inplanes * key_ratio)
            self.v_channel = inplanes - self.k_channel
        else:
            self.k_channel = inplanes
            self.v_channel = inplanes

        assert self.k_channel % head_num == 0
        assert self.v_channel % head_num == 0

        self.groups = head_num if head_split else 1
        self.query = nn.Conv2d(self.k_channel, head_num * outplanes, kernel_size=1, groups=self.groups,
                               stride=1, padding=0)
        self.value = nn.Conv2d(self.v_channel, head_num * outplanes, kernel_size=1, groups=self.groups,
                               stride=1, padding=0)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.flatten(x, 2).unsqueeze(dim=-2)
        N, C, _, L = x.shape
        if self.kv_split:
            a = torch.softmax(self.query(x[:, :self.k_channel]).reshape(N, self.head_num, -1, L), dim=3)
            v = self.value(x[:, self.k_channel:]).reshape(N, self.head_num, -1, L)
        else:
            a = torch.softmax(self.query(x).reshape(N, self.head_num, -1, L), dim=3)
            v = self.value(x).reshape(N, self.head_num, -1, L)
        v = self.dropout(v)
        x = torch.sum(a * v, dim=(1, 3))
        return x

