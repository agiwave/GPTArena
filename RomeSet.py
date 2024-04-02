import aka.nn as nn
import aka.numpy as np

def RomeSetArgs(name):
    mlp_args = dict(
        name = 'MLP',
        qk_dim = 384,
        kv_size = 384 * 3,
        kv_gate = False,
        RWKV_Ver = None
    )
    attn_args = dict(
        name = 'Attention',
        windows_size = 128,
        num_heads = 8,
        d_state = 1,
        num_kv_groups = 8,
        rotary_embedding = True,
        conv_kernel_size = 4
    )
    args = dict(
        vocab_dim = 32,
        latent_dim = 384,
        layers = [attn_args, mlp_args]*8,
        resident_gate = True,
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    match name:
        case 'vsMambaOnly':
            args['layers'] = [dict(
                name = 'Mamba',
                num_heads = 8,
                d_state = 1,
                conv_kernel_size = 4
            )]*len(args['layers'])
        case 'mambamlp':
            attn_args['name'] = 'Mamba'
        case 'mambaret':
            args['layers'] = [
                dict(
                    name = 'Mamba',
                    num_heads = 8,
                    d_state = 1,
                    conv_kernel_size = 4
                ),
                dict(
                    name = 'Retention',
                    windows_size = 128,
                    num_heads = 8,
                    num_kv_groups = 8,
                    rotary_embedding = True,
                    conv_kernel_size = 4
                )
            ]*(len(args['layers'])//2)
        case 'mambaatt':
            args['layers'] = [
                dict(
                    name = 'Mamba',
                    num_heads = 8,
                    d_state = 1,
                    conv_kernel_size = 4
                ),attn_args
            ]*(len(args['layers'])//2)
        case 'vsbase':
            mlp_args['qk_dim'] = mlp_args['qk_dim']
        case 'vsvocabFull':
            args['vocab_dim'] = args['latent_dim']
        case 'vsvocab16':
            args['vocab_dim'] = 16
        case 'vsqk_dim':
            mlp_args['qk_dim'] = 64
        case 'vskv_gate':
            mlp_args['kv_gate'] = True
        case 'vsresident_scale':
            args['resident_gate'] = True
        case 'vsHawk':
            attn_args['name'] = 'Hawk'
        case 'vsHawkOnly':
            attn_args['name'] = 'Hawk'
            args['layers'] = [attn_args]*len(args['layers'])
        case 'vsHawkOnlyH':
            attn_args['name'] = 'Hawk'
            attn_args['num_heads'] = args['latent_dim']
            args['layers'] = [attn_args]*len(args['layers'])
        case 'vsSSM':
            attn_args['name'] = 'SSM'
        case 'vsHawkRWKVCMixer':
            attn_args['name'] = 'Hawk'
            mlp_args['name'] = 'RWKVCMixer'
        case 'vsAFT':
            attn_args['name'] = 'AFT'
        case 'vsRet':
            attn_args['name'] = 'Retention'
        case 'vsRetRWKVCMixer':
            attn_args['name'] = 'Retention'
            mlp_args['name'] = 'RWKVCMixer'
        case 'vsBaseRWKVCMixer':
            mlp_args['name'] = 'RWKVCMixer'
        case 'vsRetlr':
            attn_args['name'] = 'Retention'
            attn_args.lr = True
        case 'vsRetRKWV':
            attn_args['name'] = 'RWKVTMixer'
            mlp_args['name'] = 'RWKVCMixer'
        case 'vsTopk':
            mlp_args['activation'] = 'topk'
        case 'vsBias':
            args.bias = True
        case 'Ret15m':
            attn_args['name'] = 'Retention'
            args['layers'] = [attn_args, mlp_args]*11
        case 'AFT15m':
            attn_args['name'] = 'AFT'
            args['layers'] = [attn_args, mlp_args]*12
        case 'Gemma15m':
            args['layers'] = [attn_args, mlp_args]*12
        case '20m':
            args['layers'] = [attn_args, mlp_args]*15
        case '70m':
            args.update(dict(
                layers = [attn_args, mlp_args]*30,
                latent_dim = 512,
                num_heads = 8,
                num_kv_groups = 8,
                kv_size = 512*3,
            ))

        case _:
            assert False, f"Unknown Gemma name{name}"
    return args

if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    roles = [
        # 'RomeSet-vsbase',
        # 'RomeSet-vsvocabFull',
        # 'RomeSet-vsqk_dim',
        # 'RomeSet-vskv_gate',
        # 'RomeSet-vsAFT',
        # 'RomeSet-vsHawk',
        # 'RomeSet-vsHawkOnly',         # TOP 1
        'RomeSet-vsHawkOnlyH',
        # 'RomeSet-vsHawkAtt',
        # 'RomeSet-vsSSM',
        # 'RomeSet-vsHawkRWKVCMixer',
        # 'RomeSet-vsRetRWKVCMixer',
        # 'RomeSet-vsBaseRWKVCMixer',
        # 'RomeSet-vsRet',
        # 'RomeSet-vsRetRKWV',
        # 'RomeSet-vsMambaOnly',
        # 'RomeSet-mambaatt',
        # 'RomeSet-mambamlp',
        # 'RomeSet-mambaret',
        # 'RomeSet-vsRetlr',
        # 'RomeSet-vsvocab16',          # 200321 - (-4)
        # 'RomeSet-vsresident_scale',   # add score a little bit
        # 'RomeSet-vssum_scale',        # 200321 - (-1)
        # 'RomeSet-vsTopk',             # 200321 - (-2)
        # 'RomeSet-vsBias',             # 200321 - (-3)

        # 'RomeSet-Gemma15mTopk',
        # 'RomeSet-Gemma15m',
        # 'RomeSet-AFT15m',
        # 'RomeSet-Ret15m',
        # 'RomeSet-Gemma15mNOV',
    ]
    TrainRoles(roles, lr = 6e-3, epochs=1)
    # RunRoles(roles, 'My lord Sebastian')
