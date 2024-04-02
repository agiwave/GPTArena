
import aka.nn as nn
import aka.numpy as np

def AFTBlock(**kwargs):
    return AFTFullBlock(**kwargs)

# https://blog.csdn.net/wizardforcel/article/details/132206172
def AFTFullBlock(**kwargs):
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full

        Number of heads is 1 as done in the paper
        '''
        latent_dim = args.latent_dim
        # -- Attention Args
        # args = args.attn_args
        attn_qk_dim = getattr(args, 'qk_dim', latent_dim)
        window_size = getattr(args, 'window_size', 1024)

        dim = latent_dim
        hidden_dim=attn_qk_dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.window_size = window_size
        self.wbias = nn.Parameter(shape=(window_size, window_size), initializer='xavier_uniform')
        return self

    def forward(self, x, state=None, **kwargs):
        B, T, _ = x.shape
        assert T <= self.window_size
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)
        temp_wbias = self.wbias[:T, :T].unsqueeze(0) # sequences can still be variable length

        '''
        From the paper
        '''
        Q_sig = np.sigmoid(Q)
        # temp = np.exp(temp_wbias) @ np.mul(np.exp(K), V)
        # weighted = temp / (np.exp(temp_wbias) @ np.exp(K))
        exp_K = np.exp(K)
        exp_wbias = np.exp(temp_wbias)
        weighted = (exp_wbias @ np.mul(exp_K, V)) / (exp_wbias @ exp_K)
        Yt = np.mul(Q_sig, weighted)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)
        return Yt

    return __init__(nn.Module(forward=forward), **kwargs)

def AFTSimple(args):
    def __init__(self, args):
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        
        Number of Heads is 1 as done in the paper.
        '''
        latent_dim = args.latent_dim
        # args = args.attn_args
        attn_qk_dim = getattr(args, 'qk_dim', latent_dim)
        attn_window_size = getattr(args, 'window_size', 256)

        max_seqlen = attn_window_size
        dim = latent_dim
        hidden_dim=attn_qk_dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)

    def forward(self, x, **kwargs):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)

        '''
        From the paper
        '''
        weights = np.mul(np.softmax(K, 1), V).sum(dim=1, keepdim=True)
        Q_sig = np.sigmoid(Q)
        Yt = np.mul(Q_sig, weights)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)
        return Yt

    return __init__(nn.Module(forward=forward), nn.Object(**kwargs))

def AFTLocal(args):
    def __init__(self, args):
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        s: the window size used for AFT-Local in the paper

        Number of heads is 1 as done in the paper
        '''
        latent_dim = args.latent_dim
        # args = args.attn_args
        attn_qk_dim = getattr(args, 'qk_dim', latent_dim)
        attn_window_size = getattr(args, 'window_size', 256)

        max_seqlen = attn_window_size
        dim = latent_dim
        hidden_dim=attn_qk_dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.wbias = nn.Parameter(shape=(max_seqlen, max_seqlen), initializer='xavier_uniform')
        self.max_seqlen = max_seqlen
        self.s = s

    def forward(self, x, **kwargs):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)
        self.wbias = nn.Parameter(nn.array([
            [self.wbias[i][j] if math.fabs(i-j) < self.s else 0 for j in range(self.max_seqlen)] 
            for i in range(self.max_seqlen)
            ]))
        temp_wbias = self.wbias[:T, :T].unsqueeze(0) # sequences can still be variable length

        '''
        From the paper
        '''
        Q_sig = np.sigmoid(Q)
        # temp = np.exp(temp_wbias) @ np.mul(np.exp(K), V)
        # weighted = temp / (np.exp(temp_wbias) @ np.exp(K))
        exp_K = np.exp(K)
        exp_wbias = np.exp(temp_wbias)
        weighted = (exp_wbias @ np.mul(exp_K, V)) / (exp_wbias @ exp_K)
        Yt = np.mul(Q_sig, weighted)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)
        return Yt

    return __init__(nn.Module(forward=forward), nn.Object(**kwargs))

if __name__ == "__main__":
    from RomeArena import TrainRoles
    TrainRoles([
        'RomeSet-AFT15m'
    ], lr = 6e-4, epochs=3)
