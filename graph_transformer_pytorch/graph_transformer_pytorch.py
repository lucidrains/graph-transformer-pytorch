import torch
from torch import nn, einsum
from einops import rearrange, repeat

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

List = nn.ModuleList

# normalizations

class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args,**kwargs)

# gated residual

class Residual(nn.Module):
    def forward(self, x, res):
        return x + res

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        pos_emb = None,
        dim_head = 64,
        heads = 8,
        edge_dim = None
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.pos_emb = pos_emb

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edges, mask = None):
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim = -1)

        e_kv = self.edges_to_kv(edges)

        q, k, v, e_kv = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h = h), (q, k, v, e_kv))

        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(nodes.shape[1], device = nodes.device))
            freqs = rearrange(freqs, 'n d -> () n d')
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        ek, ev = e_kv, e_kv

        k, v = map(lambda t: rearrange(t, 'b j d -> b () j d '), (k, v))
        k = k + ek
        v = v + ev

        sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') & rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h = h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b i j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# optional feedforward

def FeedForward(dim, ff_mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        nn.GELU(),
        nn.Linear(dim * ff_mult, dim)
    )

# classes

class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        edge_dim = None,
        heads = 8,
        gated_residual = True,
        with_feedforwards = False,
        norm_edges = False,
        rel_pos_emb = False,
        accept_adjacency_matrix = False
    ):
        super().__init__()
        self.layers = List([])
        edge_dim = default(edge_dim, dim)
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()

        self.adj_emb = nn.Embedding(2, edge_dim) if accept_adjacency_matrix else None

        pos_emb = RotaryEmbedding(dim_head) if rel_pos_emb else None

        for _ in range(depth):
            self.layers.append(List([
                List([
                    PreNorm(dim, Attention(dim, pos_emb = pos_emb, edge_dim = edge_dim, dim_head = dim_head, heads = heads)),
                    GatedResidual(dim)
                ]),
                List([
                    PreNorm(dim, FeedForward(dim)),
                    GatedResidual(dim)
                ]) if with_feedforwards else None
            ]))

    def forward(
        self,
        nodes,
        edges = None,
        adj_mat = None,
        mask = None
    ):
        batch, seq, _ = nodes.shape

        if exists(edges):
            edges = self.norm_edges(edges)

        if exists(adj_mat):
            assert adj_mat.shape == (batch, seq, seq)
            assert exists(self.adj_emb), 'accept_adjacency_matrix must be set to True'
            adj_mat = self.adj_emb(adj_mat.long())

        all_edges = default(edges, 0) + default(adj_mat, 0)

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes, all_edges, mask = mask), nodes)

            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)

        return nodes, edges
