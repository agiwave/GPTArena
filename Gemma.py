import aka.nn as nn
import aka.numpy as np

def GemmaEmbNorm():
    def forward(self, x):
        return x * (x.size(-1)**0.5)
    return nn.Module(
        forward = forward
    )

def GemmaArgs(name):
    class Tokenizer:
        def __init__(self, path):
            from sentencepiece import SentencePieceProcessor
            self.tokenizer = SentencePieceProcessor('data/Gemma/tokenizer.model')
            self.bos_token_id = self.tokenizer.bos_id()
            self.eos_token_id = self.tokenizer.eos_id()
        def encode(self, s):
            return self.tokenizer.encode(s)
        def decode(self, s):
            return self.tokenizer.decode(s)

    mlp_args = dict(
        name = "MLP",
        k_size= 0,
        kv_gate = True,
    )
    attn_args = dict(
        name = 'Attention',
        windows_size = 256,  # Limit Attention Seq Length to 256. Gemma2b --> 8192
        num_heads = 8,
        num_kv_groups = 1,
        rotary_embedding = True,
    )
    args = dict(
        tokenizer = Tokenizer('data/Gemma/tokenizer.model'),
        vocab_size = 256000,
        latent_dim = 2048,
        prev_norm = 'gemma',
        layers = [attn_args, mlp_args]*18,
        dropout = 0.0,
        bias = False, # bias in Linear?
    )
    match name:
        case '2b':
            args['latent_dim'] = 2048
            attn_args['num_heads'] = 8
            attn_args['num_kv_groups'] = 1
            mlp_args.k_size= 16384
            args['layers'] = [attn_args, mlp_args]*18
        case '8b':
            n_layers = 28
            args['latent_dim'] = 3072
            attn_args['num_heads'] = 16
            attn_args['num_kv_groups'] = 16
            attn_args['k_size']
            args['layers'] = [attn_args, mlp_args]*28
        case '20m':
            n_layers = 10
            args['latent_dim'] = 384
            attn_args['num_heads'] = 6
            attn_args['num_kv_groups'] = 6
            attn_args.window_size = 256
            mlp_args.k_size= 1024
            args['layers'] = [attn_args, mlp_args]*10
        case '70m':
            n_layers = 20
            args['latent_dim'] = 512
            attn_args['num_heads'] = 8
            attn_args['num_kv_groups'] = 8
            attn_args.window_size = 256
            mlp_args.k_size= 512*3
            args['layers'] = [attn_args, mlp_args]*20
        case _:
            assert False, f"Unknown Gemma name{name}"
    return args

def Gemma(name, ckpt=None):
    from CausalLM import CausalLM
    m = CausalLM(**GemmaArgs(name))
    if ckpt is not None:
        state = np.load(
            ckpt, mmap=True, weights_only=True,
        )['model_state_dict']

        '''
        Notice: All RMSNorm weight in gemma shoud be added by 1 first. See reason below(in Gemma):
            def RMSNorm(dim: int, eps: float = 1e-6, add_unit_offset: bool = True):
                def forward(self, x):
                    x = (x.float() * np.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
                    if self.add_unit_offset:
                        output = x * (1 + self.weight) # Look here :)
                    else:
                        output = x * self.weight
                    return output
                return nn.Module(
                    forward = forward,
                    eps = eps,
                    add_unit_offset = add_unit_offset,
                    weight = nn.Parameter(np.ones(dim)))
        ''' 
        with np.no_grad():
            m.embedding.weight.copy_(state['embedder.weight'])
            m.post_norm.weight.copy_(state['model.norm.weight']+1)
            for i in range(len(m.layers)//2):
                m.layers[i*2].norm.weight.copy_(state[f'model.layers.{i}.input_layernorm.weight']+1)
                m.layers[i*2].layer.in_proj.weight.copy_(state[f'model.layers.{i}.self_attn.qkv_proj.weight'])
                m.layers[i*2].layer.out_proj.weight.copy_(state[f'model.layers.{i}.self_attn.o_proj.weight'])
                m.layers[i*2+1].norm.weight.copy_(state[f'model.layers.{i}.post_attention_layernorm.weight']+1)
                m.layers[i*2+1].layer.gate_proj.data.copy_(state[f'model.layers.{i}.mlp.gate_proj.weight'])
                m.layers[i*2+1].layer.up_proj.data.copy_(state[f'model.layers.{i}.mlp.up_proj.weight'])
                m.layers[i*2+1].layer.down_proj.data.copy_(state[f'model.layers.{i}.mlp.down_proj.weight'])
    return m

if __name__ == "__main__":
    m = Gemma('2b', 'data/Gemma/gemma-2b-it.ckpt')
    print('Model loaded')
    for w in m.generator("The life meaning is"):
        print(w, end='')
