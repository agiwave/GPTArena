import aka.nn as nn
import aka.numpy as np

def HawkBlock(**kwargs):
    '''
    Paper: Griffin & Hawk
    Changes to paper:
        1, Add num_heads to RG_LRU. The orginal paper is element-wise RG_LRU (num_heads==D)
        2, Add silu after conv.
        3, beta = 1 - alpha. The orginal paper is beta = sqrt(1-alpha**2)
    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.v_gate_dim = self.hidden_dim if getattr(args, 'v_gate', False) else 0
        self.o_gate_dim = args.latent_dim if getattr(args, 'o_gate', True) else 0
        self.num_heads = getattr(args, 'num_heads', 8)
        assert self.hidden_dim % self.num_heads == 0
        # rg, ig, v, vg, og
        self.in_proj = nn.Linear(args.latent_dim, self.num_heads*2 + self.hidden_dim + self.v_gate_dim + self.o_gate_dim, bias=args.bias)
        self.conv_kernel_size = getattr(args, 'conv_kernel_size', 4)
        self.prev_conv = getattr(args, 'prev_conv', True)
        self.post_conv = getattr(args, 'post_conv', False)
        self.conv1d = None if not (self.prev_conv or self.post_conv) else nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            bias=args.bias,
            kernel_size=self.conv_kernel_size,
            groups=self.hidden_dim,
            padding=0
        )
        self.c = nn.Parameter(np.array(-8.), requires_grad=False)
        self.delta = nn.Parameter(np.array(0.5))
        self.out_proj = nn.Linear(self.hidden_dim, args.latent_dim, bias=args.bias)
        return self

    def conv(x, k, kernel_size, state, key):
        (b, l, d) = x.shape
        x = np.rearrange('b l d->b d l', x)
        if state is not None:
            conv_state = state.get(key,None)
            if conv_state is not None:
                x = np.cat((state[key], x), dim=2)
            state[key] = x[:, :, (1 - kernel_size):].detach()
        if x.size(2) < l + kernel_size - 1:
            x = np.pad(x, (l + kernel_size - 1 - x.size(2), 0), mode='replicate')
        x = k(x)
        x = np.silu(x)
        return np.rearrange('b d l->b l d', x)

    def forward(self, x, state=None, **kwargs):
        (b, l, d) = x.shape
        (rg, ig, x, vg, og) = self.in_proj(x).split([self.num_heads, self.num_heads, self.hidden_dim, self.v_gate_dim, self.o_gate_dim], dim=-1)
        
        # -- Prev Conv -- 
        x = x if not self.prev_conv else conv(x, self.conv1d, self.conv_kernel_size, state, 'prev_conv_state')

        # -- RG_LRU or GRU --
        rg = np.sigmoid(rg)
        rg = ((self.c * np.softplus(self.delta)) * rg).unsqueeze(-1)  # [B,L,H]

        x = np.rearrange('b l (h d)->b l h d', x, h=self.num_heads) # [B,L,H,D]
        x = (1-np.exp(rg)) * np.sigmoid(ig).unsqueeze(-1) * x # The orginal paper: np.sqrt(1-rg**2)*np.sigmoid(ig).unsqueeze(-1) * x
        gru_state = None if state is None else state.get('gru_state',None)
        gru_state = gru_state if gru_state is not None else np.zeros(b, 1, self.num_heads, self.hidden_dim//self.num_heads, device=x.device)

        # ---- RNN --->
        if True: # Trunc-Wise Implementation, Walk around for L*L complexity.
            (begin, step) = (0, 128)
            mask = np.tril(np.ones(step, step, device=x.device))[:,:,None,None]   #[l,h,d]
            while begin < l:
                end = begin + step if l-begin>step else l
                maskA = mask[:end-begin,:end-begin]
                truncA, truncX = [item[:, begin:end] for item in [rg, x]]
                cumA = truncA.unsqueeze(2) * maskA                   #[b,l,1,h,d]
                cumA = np.exp(np.cumsum(cumA, dim=1)) * maskA        #[b,l,m,h,d]
                shiftB = np.cat([gru_state, truncX[:,:end-begin-1]], dim=1)
                x[:,begin:end] = np.einsum('blmhd,bmhd->blhd', cumA, shiftB) + truncX
                gru_state = x[:,end-1:end]
                begin = end
        elif False: # Approximate version and faster. May cause vanishing gradient
            cumA = np.exp(np.cumsum(rg, dim=1))
            shiftA = np.pad(cumA, (0, 0, 0, 0, 1, -1), value=1.0)
            shiftB = np.cat([gru_state, x[:,:l-1]], dim=1) / (1e-10+shiftA)
            x = np.einsum('blhd,lm,bmhd->blhd', cumA, mask, shiftB) + x
        # <--- RNN ----

        if state is not None:
            state['gru_state'] = x[:,-1:].detach()
        x = np.rearrange('b l h d->b l (h d)',x)

        # -- Post Conv -- 
        x = x if not self.post_conv else conv(x, self.conv1d, self.conv_kernel_size, state, 'post_conv_state')

        # Gate and Output
        x = x if self.v_gate_dim <= 0 else x * np.gelu(vg)
        return self.out_proj(x) if self.o_gate_dim <=0 else self.out_proj(x) * np.gelu(og)
    return __init__(nn.Module(forward = forward),**kwargs)

def HawkArgs(name):
    args = dict(
        vocab_dim = 32,
        latent_dim = 384,
        layers = [dict(
            name = 'Hawk',
            num_heads = 8,
            conv_kernel_size = 4
        )]*16,
        resident_gate = True,
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    match(name):
        case 'Hawk':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'Hawk',
                        num_heads = 8,
                    ),
                    dict(
                        name = "MLP",
                        k_size = args['latent_dim']*3,
                        kv_gate = True
                    )
                ]*8,
            )
        case 'HawkOnly':
            return dict(
                args,
                layers = [dict(
                    name = 'Hawk',
                    num_heads = 8
                )]*48,
            )
        case 'Griffin':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'Attention',
                        num_heads = 8,
                        num_kv_groups = 8,
                        rotary_embedding = True,
                    ),
                    dict(
                        name = 'Hawk',
                        num_heads = 8,
                    ),
                ]*8,
            )
        case 'Mamba':
            return dict(
                args,
                layers = [dict(
                    name = 'Mamba',
                    num_heads = 8,
                )]*16,
            )
        case 'RWKV':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'RWKVTMixer',
                        num_heads = 8,
                    ),
                    dict(
                        name = 'RWKVCMixer',
                        k_size = args['latent_dim']*3,
                        kv_gate = True
                    )
                ]*8,
            )
        case 'SSMOnly':
            return dict(
                args,
                layers = [dict(
                    name = 'SSM',
                    num_heads = 8,
                )]*16,
            )
        case _:
            assert False

if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    roles = [
        # 'Hawk-Hawk',
        # 'Hawk-Mamba',
        # 'Hawk-Griffin',
        # 'Hawk-RWKV',
        'Hawk-HawkOnly',
        # 'Hawk-SSMOnly',
    ]
    TrainRoles(roles, lr = 6e-3, epochs=10)
    # RunRoles(roles, 'My lord Sebastian')
