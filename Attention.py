import aka.nn as nn
import aka.numpy as np

try:
    from xformers.ops.fmha import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import LowerTriangularFromBottomRightMask
except ImportError:
    memory_efficient_attention = None
    LowerTriangularFromBottomRightMask = None

def AttentionBlock(**kwargs):
    '''
    Group-Query Attention
    Args:
        args.latent_dim 
        args.attn_args.qk_dim(Optional, default: latent_dim)

    Examples:
        default ==> Attention
        args.attn_args.attn_heads = 8 ==> MHA: Multi-Head Attention
        args.attn_args.kv_groups = 1 ==> MQA: Multi-Query Attention
        args.attn_args.kv_groups = 2 ==> GQA: Group-Query Attention
    '''
    def __init__(self,args):
        # -- Global Args --
        latent_dim = args.latent_dim
        bias = getattr(args, 'bias', False)
        dropout = getattr(args, 'dropout', 0.2)

        # -- Attention Args
        # args = args.attn_args
        attn_qk_dim = getattr(args, 'qk_dim', latent_dim)
        attn_hidden_dim = getattr(args, 'hidden_dim', latent_dim)
        attn_heads = getattr(args, 'num_heads', 1)
        attn_kv_groups = getattr(args, 'num_kv_groups', attn_heads)
        attn_head_dim = attn_qk_dim//attn_heads
        k_dim = attn_head_dim * attn_kv_groups
        v_dim = attn_hidden_dim//attn_heads * attn_kv_groups
        assert attn_head_dim * attn_heads == attn_qk_dim
        assert attn_heads % attn_kv_groups == 0
        assert attn_hidden_dim % attn_heads == 0
        self.in_proj = nn.Linear(latent_dim, attn_qk_dim + k_dim + v_dim, bias=bias)
        self.out_proj = nn.Linear(attn_hidden_dim, latent_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.window_size = getattr(args, 'window_size', None)
        self.hidden_dim = attn_hidden_dim
        self.attn_qk_dim = attn_qk_dim
        self.attn_k_dim = k_dim
        self.attn_v_dim = v_dim
        self.attn_heads = attn_heads
        self.attn_kv_groups = attn_kv_groups
        self.scale_dk = attn_head_dim**-0.5
        self.rot_embedding = getattr(args, 'rotary_embedding', False)
        self.rope_theta = getattr(args, 'rope_theta', 10000)
        return self

    def apply_rotary_emb(x, cache, pos=0):
        '''
        Reference: LLaMA and Gemma
        Applies the rotary embedding to the query and key.
        '''
        B,L,N,D = x.shape
        slen = pos+L
        freqs_cis = cache.get('freqs_cis', None)
        if freqs_cis is None or len(freqs_cis) < slen:
            """Precomputes the frequency cis."""
            freqs = 1.0 / (10000**(np.arange(0, D, 2)[:(D // 2)].float() / D))
            t = np.arange(slen, device=freqs.device)
            freqs = np.outer(t, freqs).float()
            freqs_cis = np.polar(np.ones_like(freqs), freqs)  # complex64
            cache['freqs_cis'] = freqs_cis

        y = np.reshape(x, (B,L,N,2,D//2)).float()
        y = np.einsum('blncd->bnldc',y)
        y = np.view_as_complex(y.contiguous())
        y = np.view_as_real(y*freqs_cis[pos:pos+L]).type_as(x)
        y = np.einsum('bnldc->blncd', y)
        return np.reshape(y, (B,L,N,D))

    def causal_mask(shape, dtype, *, window_size = None, from_bottomright: bool = False,):
        mask = np.full(shape, dtype=dtype, fill_value=1)
        shift = 0 if not from_bottomright else shape[-1] - shape[-2] # q_len - k_len
        mask = np.tril(mask, diagonal=shift)
        if window_size is not None:
            mask = np.triu(mask, diagonal=shift - window_size + 1)
        return np.log(mask)

    def forward(self, x, *, cache={}, state=None, **kwargs):
        B, L, _ = x.size()

        # -- qkv --
        attn_heads, attn_kv_groups = self.attn_heads, self.attn_kv_groups
        q, k, v  = self.in_proj(x).split([self.attn_qk_dim, self.attn_k_dim, self.attn_v_dim], dim=2)
        q = q.view(B, L, attn_heads, -1)
        k = k.view(B, L, attn_kv_groups, -1)
        v = v.view(B, L, attn_kv_groups, -1)

        # -- append kv cache --
        if state is not None:
            window_size = self.window_size if self.window_size is not None else 128
            if 'cache_kv' in state:
                cache_k, cache_v = state['cache_kv']
                k = np.cat((cache_k, k), dim=1)
                v = np.cat((cache_v, v), dim=1)
            state['cache_kv'] = (k[:,1-window_size:].detach(), v[:,1-window_size:].detach())

        # -- rotary embedding --
        M = k.size(1)
        if self.rot_embedding:
            q = apply_rotary_emb(q, cache, M-L)
            k = apply_rotary_emb(k, cache)

        # -- repeat kv to match q, MQA and GQA --
        if attn_kv_groups != attn_heads:
            # [B, L, N, D]
            k = np.repeat(k, attn_heads // attn_kv_groups, dim=2)
            v = np.repeat(v, attn_heads // attn_kv_groups, dim=2)

        # -- attn --
        if memory_efficient_attention is not None:
            if self.window_size is None:
                y = memory_efficient_attention(q,k,v, attn_bias=LowerTriangularFromBottomRightMask())
            else:
                y = memory_efficient_attention(q,k,v, attn_bias=LowerTriangularFromBottomRightLocalAttentionMask(self.window_size))
        else:
            att = np.einsum('blnd,bmnd->bnlm', q, k) * self.scale_dk
            att = att + causal_mask((L,M), q.dtype, window_size=self.window_size, from_bottomright=True)
            att = np.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = np.einsum('bnlm,bmnd->blnd', att, v)
        y = y.reshape(B, L, self.hidden_dim)
        return self.resid_dropout(self.out_proj(y))
    return __init__(nn.Module(forward=forward), nn.Object(**kwargs))

# --- Example ---
if __name__ == "__main__":
    atten = AttentionBlock(
        latent_dim = 384,
        window_size = 128,
        hidden_dim = 256,
        qk_dim = 384,
        num_heads = 8,
        num_kv_groups = 2
    )
    input = np.randn(50, 100, 384)
    output = atten(input)
    print(output.size())