
import aka.nn as nn
import aka.numpy as np

def RWKVTMixerBlock(**kwargs):
    return RWKV_Tmix_x060(**kwargs)

def RWKV_Tmix_x060(**kwargs):
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        self.dim_att = getattr(args, 'dim_att', args.latent_dim)
        self.n_head = args.num_heads # dim_att // self.head_size
        assert self.dim_att % self.n_head == 0

        with np.no_grad():
            layer_id = 5
            n_layers = 10
            ratio_0_to_1 = layer_id / (n_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layers)  # 1 to ~0
            ddd = np.ones(1, 1, args.latent_dim)
            for i in range(args.latent_dim):
                ddd[0, 0, i] = i / args.latent_dim

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - np.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - np.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - np.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (np.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - np.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - np.pow(ddd, 0.5 * ratio_1_to_almost0))

            TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(np.zeros(args.latent_dim, TIME_MIX_EXTRA_DIM*5).uniform_(-1e-4, 1e-4))
            self.time_maa_w2 = nn.Parameter(np.zeros(5, TIME_MIX_EXTRA_DIM, args.latent_dim).uniform_(-1e-4, 1e-4))

            # fancy time_decay
            decay_speed = np.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -6 + 5 * (n / (self.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.dim_att))

            TIME_DECAY_EXTRA_DIM = 64
            self.time_decay_w1 = nn.Parameter(np.zeros(args.latent_dim, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_decay_w2 = nn.Parameter(np.zeros(TIME_DECAY_EXTRA_DIM, self.dim_att).uniform_(-1e-4, 1e-4))

            tmp = np.zeros(self.dim_att)
            for n in range(self.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (self.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.dim_att//self.n_head))

        self.receptance = nn.Linear(args.latent_dim, self.dim_att, bias=False)
        self.key = nn.Linear(args.latent_dim, self.dim_att, bias=False)
        self.value = nn.Linear(args.latent_dim, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, args.latent_dim, bias=False)
        self.gate = nn.Linear(args.latent_dim, self.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, self.dim_att, eps=(1e-5)*(getattr(args,'head_size_divisor',8)**2))
        return self

    def forward(self, x, state=None, **kwargs):
        B, T, C = x.size()

        xx = np.pad(x, (0,0,1,-1), value=0.) - x

        xxx = x + xx * self.time_maa_x
        xxx = np.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = np.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = np.silu(self.gate(xg))

        ww = np.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        if False:# TODO translate CUDA to torch
            x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)
        else:
            k = np.clamp(k, max=30, min=-60) # clamp extreme values. e^30 = 10^13
            k = np.exp(k)
            sum_k = np.cumsum(k, dim=1)
            kv = (k * v).view(B, T, self.n_head, -1)
            w = w.view(B, T, self.n_head, -1)
            k = k.view(B, T, self.n_head, -1)
            # wkv = (np.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)
            mask = np.tril(np.ones(T,T, device=x.device))
            wkv = (np.einsum('bthc,tu,buhc->bthc', w, mask, kv)).contiguous().view(B, T, -1)
            wk = (np.einsum('bthc,tu,buhc->bthc', w, mask, k)).contiguous().view(B, T, -1)
            x = np.sigmoid(r) * wkv / wk
            
        x = np.reshape(x, (B * T, C))
        x = self.ln_x(x).view(B, T, C)
        return self.output(x * g)
    return __init__(nn.Module(forward=forward), **kwargs)
