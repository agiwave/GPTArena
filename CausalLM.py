import math
import aka.nn as nn
import aka.numpy as np

def MetaLayer(**kwargs):
    '''
    Build resident meta layer by name. Include: GQA(Group-Query Attention), MLP, GateMLP, ...
    '''
    def __init__(self, name, **kwargs):
        import importlib
        module = importlib.import_module(name)
        short_name = name.split('./\\')[-1]
        m = getattr(module, short_name+"Block", None)
        assert m is not None, f"Unknown layer:{name}"
        self.norm = nn.RMSNorm(kwargs['latent_dim'])
        self.layer = m(**kwargs)
        self.x_gate = None if not kwargs.get('x_gate',False) else nn.Parameter(np.ones(kwargs['latent_dim']))
        self.resident_gate = None if not kwargs.get('resident_gate',False) else nn.Parameter(np.ones(kwargs['latent_dim']))
        return self

    def forward(self, x, **kwargs):
        y = self.norm(x)
        if self.x_gate is not None:
            x_gate = np.gelu(self.x_gate)
            y = self.layer(y, **kwargs)
            y = y * x_gate
        else:
            y = self.layer(y, **kwargs)
        if self.resident_gate is not None:
            x = x * np.gelu(self.resident_gate)
        return x + y, None
    return __init__(nn.Module(forward = forward), **kwargs)

def CausalLM(**kwargs):
    '''
    Causal Language Model.
    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        self.tokenizer = args.tokenizer
        self.latent_dim = args.latent_dim
        self.vocab_dim = getattr(args, 'vocab_dim', args.latent_dim)
        self.train_mode = getattr(args, 'train_mode', None)
        self.in_proj = None if self.vocab_dim == self.latent_dim else nn.Linear(self.vocab_dim, self.latent_dim, bias=args.bias)
        self.out_proj = None if self.vocab_dim == self.latent_dim else nn.Linear(self.latent_dim, self.vocab_dim, bias=args.bias)
        self.pad_x = getattr(args, 'pad_x', False)
        self.embedding_scale = (None if not getattr(args,'embedding_scale',False) else math.sqrt(vocab_dim))
        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=self.vocab_dim)

        make_layer = MetaLayer if not hasattr(args, 'MetaLayer') else args.MetaLayer
        self.layers = nn.ModuleList([make_layer(**dict(layer,**kwargs)) for layer in args.layers])
        self.lm_head = None if not getattr(args, 'lm_head', False) else nn.Linear(self.vocab_dim, args.vocab_size,bias=False)

        prev_norm = getattr(args, 'prev_norm', None)
        if prev_norm is not None:
            match prev_norm:
                case 'gemma':
                    from Gemma import GemmaEmbNorm
                    prev_norm = GemmaEmbNorm()
                case _:
                    prev_norm = nn.RMSNorm(args.latent_dim)
        self.prev_norm = prev_norm
        self.post_norm = nn.RMSNorm(self.vocab_dim)
        self.cache = {}
        return self

    def forward(self, inputs, targets=None, state=None):
        # -- Embedding and layers
        x = self.embedding(inputs)

        # -- vocab_dim --> latent_dim
        if self.vocab_dim != self.latent_dim:
            if self.pad_x:
                x = np.pad(x, (self.latent_dim-self.vocab_dim,0), mode='constant', value=float(0.0))
            else:
                x = self.in_proj(x)

        if self.embedding_scale is not None:    # RetNet, nonsense :(. 
            x = x * self.embedding_scale

        # -- layers --
        if self.prev_norm is not None:
            x = self.prev_norm(x)
        if(state is not None):
            layer_states = state.get('layer_states', None)
            if layer_states is None:
                layer_states = [{} for _ in self.layers]
                state['layer_states'] = layer_states

        layer_losses = []
        for i in range(len(self.layers)):
            l_state = None if state is None else layer_states[i]
            x, loss = self.layers[i](x, cache=self.cache, state=l_state)
            if loss is not None:
                layer_losses.append(loss)

        # -- latent_dim --> vocab_dim
        if self.vocab_dim != self.latent_dim:
            if self.pad_x:
                x = np.pad(x, (self.vocab_dim-self.latent_dim,0), mode='constant', value=float(0.0))
            else:
                x = self.out_proj(x)

        if self.post_norm is not None:
            x = self.post_norm(x)

        # -- vocab_dim --> logits
        if self.lm_head is not None:
            y = self.lm_head(x)    # -- LLaMA vs embedding.weight ? --
        else:
            y = np.einsum('bld,nd->bln', x, self.embedding.weight) * (self.vocab_dim**-0.5)

        # -- logits --> output
        if(targets is not None):
            if self.train_mode is None:
                loss = np.cross_entropy(y.view(-1, y.size(-1)), targets.reshape(-1), ignore_index=-1)
                if len(layer_losses) > 0:
                    loss += np.sum(np.stack(layer_losses, dim=-1)) / len(layer_losses)
            else:
                assert False
            return y, loss
        else:
            return y

    def generator(self, prompts: str, max_length : int = 64):
        prompt_tokens = [self.tokenizer.bos_token_id]+self.tokenizer.encode(prompts) # [self.tokenizer.bos_token_id]+
        if hasattr(self, 'eval'):
            self.eval()

        with np.no_grad():
            state = {}
            cache = []
            input_token_ids = np.array([prompt_tokens])
            for _ in range(max_length):
                outputs = self(input_token_ids, state=state)
                input_token_ids = np.argmax(outputs[:,-1:,:], dim=-1)
                cache = cache + input_token_ids[0].tolist()
                if self.tokenizer.eos_token_id in input_token_ids:
                    break

                word = self.tokenizer.decode(cache)
                word_token_ids = self.tokenizer.encode(word)
                if cache[-1] == word_token_ids[-1]:
                    cache = []
                    yield word

            if len(cache)>0:
                yield self.tokenizer.decode(cache)

    def generate(self, prompts : str, max_length : int = 64):
        response = ''
        for w in self.generator(prompts,max_length):
            response += w
        return response
        
    return __init__(nn.Module(forward = forward, generate = generate, generator=generator),**kwargs)

def CausalLMArgs(name):
    mlp_args = dict(
        name = 'MLP',
        k_size = 384*4,
        kv_gate = True,
        k_dim = 384,
        hidden_dim = 384
    )
    attn_args = dict(
        name = 'Attention',
        k_dim = 384,
        hidden_dim = 384,
        num_heads = 6,
        num_kv_groups = 6,
        rotary_embedding = True,
    )
    return dict(
        vocab_size = 50304,
        vocab_dim = 64,
        block_size = 256,
        latent_dim = 384,

        dropout = 0.2,
        bias = False, # do we use bias inside LayerNorm and Linear layers?
        layers = [attn_args, mlp_args]*6,
    )

if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    TrainRoles([
        'CausalLM-demo'
    ], lr = 6e-4, epochs=3)
    # RunRoles([
    #     'CausalLM-demo'
    # ], "Paul Daniels (born 4 June 1981 in Burlington)")


# One by One
# for i in range(len(prompt_tokens)-1):
#     self(np.array([prompt_tokens[i:i+1]]), state=state)
# input_token_ids = np.array([prompt_tokens[-1:]])

# Without state
# if len(prompt_tokens) > 1:
#     self(np.array([prompt_tokens[:-1]]))
# input_token_ids = np.array([prompt_tokens])
# for _ in range(max_length):
#     outputs = self(input_token_ids)
#     output_token_ids = np.argmax(outputs[:,-1:,:], dim=-1)
#     cache = cache + output_token_ids[0].tolist()
#     if self.tokenizer.eos_token_id in input_token_ids:
#         break

#     word = self.tokenizer.decode(cache)
#     word_token_ids = self.tokenizer.encode(word)
#     if cache == word_token_ids:
#         cache = []
#         yield word

#     input_token_ids = np.cat([input_token_ids, output_token_ids], dim=1)
