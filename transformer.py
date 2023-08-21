from torch import Tensor
import torch.nn.functional as f
import numpy as np
import torch
from torch import nn

def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

class Residual_FF(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        out = self.sublayer(*tensors)
        return self.norm(tensors[0] + self.dropout(out))
        return tensors[0] + self.dropout(out)

class Residual_MH(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        out, norm = self.sublayer(*tensors)
        return self.norm(tensors[0] + self.dropout(out)), norm
        return tensors[0] + self.dropout(out), norm

class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        out = self.sublayer(*tensors)
        return tensors[0] + self.norm(self.dropout(out))


class AttentionBiasHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, dim_s: int, seq_len: int, use_cls: bool = False):  # dim_s is for static features
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
        self.b = nn.Sequential(
            nn.Linear(dim_s, 1024),
            nn.ReLU(),
            nn.Linear(1024, seq_len * seq_len),
        )
        self.agg = nn.Linear(2, 1, bias = False)
        self.agg_eps = nn.Parameter(torch.rand(1))


        self.b1 = nn.Sequential(
            nn.Linear(dim_s + 32, 128), # 64 for pos emb
            nn.ReLU(),
            nn.Linear(128, dim_q),
        )
        self.q_cpe = nn.Linear(dim_q, dim_q)
        self.k_cpe = nn.Linear(dim_q, dim_q)

        self.use_cls = use_cls
        self.dim_q = dim_q

    def forward(self, query: Tensor, key: Tensor, value: Tensor, sf: Tensor, pos : Tensor = None) -> Tensor:
        b, l, d = query.shape
        query = self.q(query.reshape(b * l, d)).reshape(b, l, -1)
        key = self.k(key.reshape(b * l, d)).reshape(b, l, -1)
        value = self.v(value.reshape(b * l, d)).reshape(b, l, -1)

        if pos is not None:  # pairwise CPE
            sf_expand = sf.unsqueeze(1).expand(b,l,sf.shape[1])
            pos = pos.unsqueeze(0).expand(b,l,pos.shape[1])
            cpe = torch.concat([sf_expand, pos], axis = 2)
            cpe = self.b1(cpe)
            query_cpe = self.q_cpe(cpe.reshape(b * l, self.dim_q)).reshape(b, l, -1)
            key_cpe = self.k_cpe(cpe.reshape(b * l, self.dim_q)).reshape(b, l, -1)
            bias = query_cpe.bmm(key_cpe.transpose(1, 2))
        else:  # absolute CPE
            bias = self.b(sf).reshape(b, l, l)

        add_mode = 0

        if add_mode == 0:
            temp = query.bmm(key.transpose(1, 2))
            scale = query.size(-1) ** 0.5
            softmax = f.softmax(temp / scale, dim=-1) + bias
            out = softmax.bmm(value)

        # no softmax at all
        if add_mode == 1:
            temp = query.bmm(key.transpose(1, 2))
            out = (temp + bias).bmm(value)

        # MOD 2: instead of add, use a linear model
        if add_mode == 2:
            temp = query.bmm(key.transpose(1, 2)).reshape(b,l,l,1)
            scale = query.size(-1) ** 0.5
            temp = temp / scale
            temp = f.softmax(temp, dim = -1)
            bias = bias.reshape(b,l,l,1)
            att = torch.cat([temp, bias], dim = 3)
            out = self.agg(att).squeeze().bmm(value)
        # return out

        # MOD 5: linear, with activation on Q, K
        if add_mode == 3:
            query = torch.nn.functional.elu(query) + 1
            key = torch.nn.functional.elu(key) + 1
            temp = query.bmm(key.transpose(1, 2)).reshape(b,l,l,1)
            out = (temp + bias).bmm(value)

        # sigmoid for bias, and epsilon for two source. 
        if add_mode == 4:
            temp = query.bmm(key.transpose(1, 2))
            scale = query.size(-1) ** 0.5
            softmax = f.softmax(temp / scale, dim=-1)
            bias = torch.sigmoid(bias)
            eps = torch.sigmoid(self.agg_eps)
            out = (eps * temp + (1-eps) * bias).bmm(value)

        if add_mode == 5:
            temp = query.bmm(key.transpose(1, 2))
            scale = query.size(-1) ** 0.5
            softmax = 0.99*f.softmax(temp / scale, dim=-1) + 0.01*bias
            out = softmax.bmm(value)


        # do not add the att from seq
        # out = bias.bmm(value)


        # calculate the A[i,j] - A[i-1, j] etc, as an extra regularizer. Do not include the [cls] token.
        # shift up does not make sense for A[0]

        # if use fill [cls] with zeros.
        if self.use_cls:
            att_seq = bias[:,1:,1:]
        else:
            att_seq = bias
        norm3 = torch.norm(att_seq[:,1:,:] - att_seq[:,:-1,:], p = 1) + torch.norm(att_seq[:,:,1:] - att_seq[:,:,:-1], p = 1)
        return out, norm3


class MultiHeadAttentionBias(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_model: int, dim_q: int, dim_k: int, dim_s: int, seq_len: int, use_cls: bool = False):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionBiasHead(dim_in, dim_q, dim_k, dim_s, seq_len, use_cls) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, sf: Tensor, pos : Tensor = None) -> Tensor:
        # return self.linear(
        #     torch.cat([h(query, key, value, sf) for h in self.heads], dim=-1)
        # )

        results = [h(query, key, value, sf, pos) for h in self.heads]
        hidden = torch.cat([result[0] for result in results], dim = -1)
        norm = torch.sum(torch.stack([result[1] for result in results]))
        return self.linear(hidden), norm


class TransformerBiasEncoderLayer(nn.Module):
    def __init__(
            self,
            dim_in: int = 256,
            dim_model: int = 256,
            num_heads: int = 6,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            dim_s: int = 19,
            seq_len: int = 30,
            diff_dim: bool = True,
            use_cls: bool = False
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual_MH(
            MultiHeadAttentionBias(num_heads, dim_in, dim_model, dim_q, dim_k, dim_s, seq_len, use_cls),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual_FF(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor, sf: Tensor, pos: Tensor = None) -> Tensor:
        # src = self.attention(src, src, src, sf)
        # return self.feed_forward(src)

        src, norm = self.attention(src, src, src, sf, pos)
        return self.feed_forward(src), norm


class TransformerBiasEncoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 2,
            dim_in: int = 19,
            dim_model: int = 256,
            num_heads: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            dim_s: int = 19,
            seq_len: int = 30,
            diff_dim: bool = True,
            use_cls: bool = False
    ):
        super().__init__()

        #        self.first_layer = TransformerBiasEncoderLayer(dim_in, dim_model, num_heads, dim_feedforward, dropout, dim_s, seq_len)
        self.first_layer = nn.Linear(dim_in, dim_model)

        self.diff_dim = diff_dim

        dim_q, dim_k = dim_model//num_heads, dim_model//num_heads
        self.first_layer1 = MultiHeadAttentionBias(num_heads, dim_in, dim_model, dim_q, dim_k, dim_s, seq_len, use_cls=use_cls)
        self.first_layer2 = feed_forward(dim_model, dim_feedforward)

        self.layers = nn.ModuleList(
            [
                TransformerBiasEncoderLayer(dim_model, dim_model, num_heads, dim_feedforward, dropout, dim_s, seq_len, use_cls = use_cls)
                for _ in range(num_layers)
            ]
        )

        self.last_layer1 = MultiHeadAttentionBias(num_heads, dim_model, dim_model, dim_q, dim_k, dim_s, seq_len, use_cls = use_cls)
        self.last_layer2 = feed_forward(dim_model, dim_feedforward)

        self.pos_emb = torch.nn.Embedding(seq_len, 32)

    def forward(self, src: Tensor, sf: Tensor) -> Tensor:
        # batch_size, seq_len, dimension = src.size(0), src.size(1), src.size(2)
        # src += position_encoding(seq_len, dimension)

        # original
        # if self.diff_dim:
        #     src = self.first_layer1(src, src, src, sf)
        #     src = self.first_layer2(src)
        # for layer in self.layers:
        #     src = layer(src, sf)
        #
        # return src

        # option1: pos is None, means use MLP to get PE directly
        pos = None
        # option 2:  
        #pos = torch.arange(src.shape[1], dtype=torch.int).cuda() #, device=device)
        #pos = self.pos_emb(pos)#.unsqueeze(0)#.expand(src.shape[0], src.shape[1], 64)

        # MOD 3:
        total_norm = 0

        # Findings: use fc as first layer can help with the inf loss. 
        batch_size, seq_len, dimension =src.size(0), src.size(1), src.size(2)


        first_layer_mode = 1

        # linear model
        if first_layer_mode == 0:
            src = self.first_layer(src.reshape(-1, dimension)).reshape(batch_size, seq_len, -1)

        # attention without residual
        if first_layer_mode == 1:
            if self.diff_dim:
                src, norm = self.first_layer1(src, src, src, sf, pos)
                src = self.first_layer2(src)
                total_norm += norm
        for layer in self.layers:
            src, norm = layer(src, sf, pos)
            total_norm += norm

        last_layer_mode = 0
        if last_layer_mode == 0:
            pass
        if last_layer_mode == 1:
            src, norm = self.last_layer1(src, src, src, sf, pos)
            src = self.last_layer2(src)
            total_norm += norm

        return src, total_norm


class TransformerBiasNet(nn.Module):
    def __init__(
            self,
            num_layers: int = 2,
            dim_in: int = 19,
            dim_model: int = 512,
            num_heads: int = 4,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
            dim_s: int = 19,
            seq_len: int = 30,
            use_cls: int = 0, # 0 for not use, 1 for zero, 2 for param
            use_id: bool = False,
    ):
        super().__init__()
        if use_cls:
            seq_len += 1
            self.cls = torch.nn.Parameter(torch.zeros(dim_in))
        if use_id:
            dim_s += 32
        self.use_cls = use_cls
        self.encoder = TransformerBiasEncoder(num_layers, dim_in, dim_model, num_heads, dim_feedforward, dropout, dim_s,
                                              seq_len, use_cls=use_cls)
        self.fc = nn.Linear(dim_model, 1)
        self.id_emb = nn.Embedding(1045, 32)
        self.use_id = use_id
        self.cls_fc = nn.Sequential(
                            nn.Linear(dim_s, dim_feedforward),
                            nn.ReLU(),
                            nn.Linear(dim_feedforward, dim_in),
                            )

    def forward(self, src: Tensor, sf: Tensor, county_id: Tensor) -> Tensor:
        if self.use_id:
            id_emb = self.id_emb(county_id)
            sf = torch.cat([sf, id_emb], axis=1)

        if self.use_cls == 1: 
            zero = torch.zeros(src.shape[0], 1, src.shape[2]).cuda().double()
            src = torch.cat([zero, src], axis=1)
        elif self.use_cls == 2:
            cls = self.cls.reshape(1,1,src.shape[2]).expand(src.shape[0], 1, src.shape[2])
            src = torch.cat([cls, src], axis=1)
        elif self.use_cls == 3:   # if use_cls == 3, use sf to calculate cls. 
            batch_size, seq_len, d = src.shape
            cls = self.cls_fc(sf).reshape(batch_size, 1, d)
            src = torch.cat([cls, src], axis=1)

        src, norm = self.encoder(src, sf)
        out = self.fc(src[:, 0, :])
        return out, norm


class TransformerBiasAttNet(nn.Module):
    def __init__(
            self,
            num_layers: int = 3,
            dim_in: int = 19,
            dim_model: int = 256,
            num_heads: int = 4,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
            dim_s: int = 19,
            seq_len: int = 30,
            use_cls: int = 0,
            use_id: bool =  False
    ):
        super().__init__()
        if use_id:
            dim_s += 32
        if use_cls:
            seq_len += 1
            self.cls = torch.nn.Parameter(torch.zeros(dim_in))
        self.encoder = TransformerBiasEncoder(num_layers, dim_in, dim_model, num_heads, dim_feedforward, dropout, dim_s,
                                              seq_len, use_cls=use_cls)

        para_k = 4
        self.fc_county_1 = nn.Linear(dim_s, seq_len * para_k)  # county dim: lon,lat,area 12 + soil 82 + fert 2 + year 1
        self.fc_county_2 = nn.Linear(seq_len * para_k, seq_len)

        self.fc = nn.Linear(dim_model, 1)
        self.id_emb = nn.Embedding(1045, 32)

        self.use_cls = use_cls
        self.use_id = use_id

        self.cls_fc = nn.Sequential(nn.Linear(dim_s, dim_feedforward),
                                    nn.ReLU(),
                                    nn.Linear(dim_feedforward, dim_in),
                                    )

    def forward(self, src: Tensor, sf: Tensor, county_id: Tensor) -> Tensor:
        if self.use_id:
            id = self.id_emb(county_id)
            sf = torch.cat([sf, id], axis=1)

        if self.use_cls == 1:
            zero = torch.zeros(src.shape[0], 1, src.shape[2]).cuda().double()
            src = torch.cat([zero, src], axis=1)
        elif self.use_cls == 2:
            cls = self.cls.reshape(1,1,src.shape[2]).expand(src.shape[0], 1, src.shape[2])
            src = torch.cat([cls, src], axis=1)
        elif self.use_cls == 3: #, use sf to calculate cls. 
            batch_size, seq_len, d = src.shape
            cls = self.cls_fc(sf).reshape(batch_size, 1, d)
            src = torch.cat([cls, src], axis=1)


        src, total_norm = self.encoder(src, sf)
        batch_size, seq_len, dim_model = src.shape

        county_att_1 = nn.functional.relu(self.fc_county_1(sf))
        county_att_2 = self.fc_county_2(county_att_1)
        norm = torch.norm(county_att_2[:,1:] - county_att_2[:,:-1], p = 1) 

        county_att_2 = county_att_2.view([-1, seq_len, 1]).expand([-1, seq_len, dim_model])

        state = torch.sum(torch.mul(src, county_att_2), dim=1)
        last_state = nn.functional.relu(state)

        out = self.fc(last_state)
        return out, norm + total_norm
