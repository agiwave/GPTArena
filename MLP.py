import aka.nn as nn
import aka.numpy as np

def Topk(n_topk, *, dim=-1):
    def forward(self, x):
        dim = self.dim
        n_topk = self.n_topk
        v, indics = np.topk(x, n_topk, dim=dim)
        v = np.select(v, dim=dim, index=n_topk-1).unsqueeze(dim=dim)
        x = np.where(x<v,float('-inf'), x)
        return np.softmax(x, dim=dim)
    return nn.Module(forward=forward, n_topk=n_topk, dim=dim)

def MLPBlock(**kwargs):
    '''
    Reference: Gemma, LLaMA
    Common ver:
        (b,l,latent_dim) --up--> (b,l,k_size, k_size) --down--> (b, l, latent_dim)
    Full ver:
        (b,l,latent_dim) --in--> (b,l,k_dim) --up--> (b,l,k_size, k_size) 
        --down--> (b, l, hidden_dim) --out--> (b,l,latent_dim)
    Args:
        args.latent_dim,    Required
        args.bias = False,  Optional(False)
        args.mlp_args.k_dim = 384,  Optional(None), any value means in_proj to k_dim first
        args.mlp_args.k_size = 384*4, Optional(latent_dim)
        args.mlp_args.kv_gate = False, Optional(False)
        args.mlp_args.hidden_dim = 384,Optional(latetn_dim)
        args.mlp_args.num_heads = 6,      # not support.
        args.mlp_args.num_kv_groups = 6,  # not support.
    Examples:
        args.mlp_args,kv_gate == True ==> GateMLP
    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        # -- Global Args --
        latent_dim = args.latent_dim
        bias = getattr(args,'bias', False)

        # -- MLP Args
        # args = args.mlp_args
        k_size = getattr(args, 'k_size', latent_dim)
        kv_gate = getattr(args, 'kv_gate', False)
        k_dim = getattr(args, 'k_dim', None)
        hidden_dim = getattr(args, 'hidden_dim', latent_dim)
        act = getattr(args, 'activation', 'gelu')
        match act:
            case 'topk':
                self.act = Topk(*getattr(args, 'activation_args', [3]))
            case _:
                self.act = getattr(np, act)
        self.latent_dim = latent_dim
        self.k_dim = k_dim if k_dim is not None else latent_dim
        self.k_size = k_size
        self.hidden_dim = hidden_dim
        self.num_heads = getattr(args, 'num_heads', 1)
        self.in_proj = None if k_dim is None else nn.Linear(latent_dim, self.k_dim, bias=bias)        # Q
        if self.num_heads == 1:
            self.up_proj = nn.Parameter(shape=(k_size, self.k_dim))                                        # K(reversed)
            self.gate_proj = None if not kv_gate else nn.Parameter(shape=(k_size, self.k_dim))             # G or mask
            self.down_proj = nn.Parameter(shape=(hidden_dim, k_size))                                  # V
        else:
            h_k_dim = self.k_dim // self.num_heads
            h_hd_dim = hidden_dim // self.num_heads
            self.up_proj = nn.Parameter(shape=(self.num_heads, k_size, h_k_dim))                                        # K(reversed)
            self.gate_proj = None if not kv_gate else nn.Parameter(shape=(self.num_heads, k_size, h_k_dim))             # G or mask
            self.down_proj = nn.Parameter(shape=(self.num_heads, h_hd_dim, k_size))    
        self.out_proj = None if hidden_dim == latent_dim else nn.Linear(hidden_dim, latent_dim, bias=bias)
        return self

    def forward(self, x, **kwargs):
        x = x if self.in_proj is None else self.in_proj(x)
        if self.num_heads == 1:
            up = np.einsum('bld,md->blm', x, self.up_proj)
            if(self.gate_proj is not None):
                gate = np.einsum('bld,md->blm', x, self.gate_proj)
                gate = gate if self.act is None else self.act(gate)
                up = gate * up
            elif self.act is not None:
                up = self.act(up)
            down = np.einsum('bld,md->blm', up, self.down_proj)
        else:
            x = np.rearrange('b l (h d) -> b l h d', x, h=self.num_heads)
            up = np.einsum('blhd,hmd->blhm', x, self.up_proj)
            if(self.gate_proj is not None):
                gate = np.einsum('blhd,hmd->blhm', x, self.gate_proj)
                gate = gate if self.act is None else self.act(gate)
                up = gate * up
            elif self.act is not None:
                up = self.act(up)
            down = np.einsum('blhd,hmd->blhm', up, self.down_proj)
            down = np.rearrange('b l h d -> b l (h d)', down)
        return down if self.out_proj is None else self.out_proj(down)
    return __init__(nn.Module(forward=forward), **kwargs)

########################### testing ##############################
def MLPArgs(name):
    mlp_args = dict(
        name = 'MLP',
        num_heads = 1,
        k_size = 384 * 3,
        kv_gate = False,
    )
    match name:
        case 'base':
            mlp_args.num_heads = 1
            mlp_args.k_size = 384 * 3
        case 'h4':
            mlp_args.num_heads = 4
            mlp_args.k_size = 384 * 3
        case 'h8':
            mlp_args.num_heads = 8
            mlp_args.k_size = 384 * 3
        case _:
            assert False, f"Unknown name{name}"
    return dict(
        vocab_dim = 32,
        latent_dim = 384,
        resident_gate = True,
        dropout = 0.1,
        bias = False, # bias in Linear?
        layers = [
            dict(
                name = 'Attention',
                windows_size = 64,  # Limit Attention Seq Length to 256. Gemma2b --> 8192
                k_dim = 384,
                num_heads = 8,
                num_kv_groups = 8,
                rotary_embedding = True,
                num_states = 64
            ), 
            mlp_args
        ]*8,
    )

if __name__ == "__main__":
    from RomeArena import TrainRoles
    TrainRoles([
        'MLP-h4',
        'MLP-h8',
        'MLP-base'
    ], lr = 6e-4, epochs=1)
