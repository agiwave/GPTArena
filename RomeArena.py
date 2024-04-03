
import os
os.environ["aka_provider_name"] = "aka.providers.torch"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import aka.nn as nn
import aka.repo as repo
import aka.data

def TrainRoles(roles, *, tokenizer=None, lr=1.e-4, epochs=1):
    # -- Tokenizer --
    repoName = 'data/pretrain'
    if tokenizer is None:
        tokenizer = repo.AutoTokenizer(repoName)
    # class Tokenizer:
    #     def __init__(self, path):
    #         from sentencepiece import SentencePieceProcessor
    #         self.tokenizer = SentencePieceProcessor('data/Gemma/tokenizer.model')
    #         self.bos_token_id = self.tokenizer.bos_id()
    #         self.eos_token_id = self.tokenizer.eos_id()
    #     def encode(self, s):
    #         return self.tokenizer.encode(s)
    #     def decode(self, s):
    #         return self.tokenizer.decode(s)
    # vocab_size = 256000
    
    # -- Roles --
    roles = [nn.Object(name=name) for name in roles]
    import importlib
    for role in roles:
        module_name, sub_name = role.name.split('-')
        module = importlib.import_module(module_name)
        args = getattr(module, module_name+'Args')(sub_name)
        args.update(dict(
            tokenizer = tokenizer,
            vocab_size = 50304,
            dropout = 0.1,
            bias = False
        ))
        if not 'vocab_dim' in args:
            args['vocab_dim'] = 64

        role.args = args
        role.persist_filename = 'data/RomeArena/'+role.name+".ckt"

    # -- Data loader
    # dataset = repo.AutoDataset("bookcorpus", token='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')
    dataset = repo.AutoDataset('text', data_dir='data/bookcorpus', split='train')
    dataloader = aka.data.TextStreamingLoader(
                    dataset, 
                    tokenizer=tokenizer, 
                    n_tokens=512,
                    batch_size=6,
                    data_mapper=lambda x:x['text'])

    # -- Train --
    def train(role, **kwargs):
        from CausalLM import CausalLM
        return nn.train(
            CausalLM(**role.args), 
            data_loader=dataloader,
            optimizer="Adam",
            optimizer_kwargs={'lr':lr},
            forward_kwargs={'state':{}},
            persist_filename = role.persist_filename,
            persist_per_batchs = 50,
            epochs=epochs)

    # -- Plot --
    m_losses = [train(r) for r in roles]
    from matplotlib import pyplot as plt
    for v in m_losses:
        plt.plot(v)
    plt.xlabel('Iterators')
    plt.ylabel('Losses')
    plt.legend([r.name for r in roles], loc='upper right')
    plt.show()

def RunRoles(names, prompt, *, tokenizer=None, ):
    # -- Tokenizer --
    if tokenizer is None:
        tokenizer = repo.AutoTokenizer('data/mamba-370m-hf')

    # -- Roles --
    roles = [nn.Object(name=name) for name in names]
    import importlib
    for role in roles:
        module_name, sub_name = role.name.split('-')
        module = importlib.import_module(module_name)
        args = getattr(module, module_name+'Args')(sub_name)
        args.update(dict(
            tokenizer = tokenizer,
            vocab_size = 50304,
            dropout = 0.1,
            bias = False
        ))
        if not 'vocab_dim' in args:
            args['vocab_dim'] = 64
        role.args = args
        role.persist_filename = 'data/RomeArena/'+role.name+".ckt"

    # -- Run --
    for role in roles:
        from CausalLM import CausalLM
        model = CausalLM(**role.args)
        nn.load_weights(model, role.persist_filename)
        print(role.name + ":")
        for w in model.generator(prompt):
            print(w, end='')
        print('')

if __name__ == "__main__":
    TrainRoles([
        # 'Gemma-20m', 
        'RomeSet-20m',
        # 'RomeSet-24vdim',
        'RomeSet-Ret',
        # 'RomeSet-32vdim',
        # 'RomeSet-64vdim',
        # 'RomeSet-vbdimpad',
        # 'RomeSet-vbdim',
        # 'RomeSet-novbdim',
    ], lr = 6e-4, epochs=1)