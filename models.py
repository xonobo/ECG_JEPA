import torch
from time_jepa_ver4 import Time_jepa

def load_encoder(model_name, ckpt_dir, leads=None):
    model_names = ['ejepa_random', 'ejepa_multiblock']
    assert model_name in model_names, f"Model name must be one of {model_names}"

    if leads is not None:
        assert model_name in ['ejepa_random', 'ejepa_multiblock'], f"Model {model_name} does not support reduced leads"

    if leads is None:
        leads = [0,1,2,3,4,5,6,7]

    params = {
        'encoder_embed_dim': 768,
        'encoder_depth': 12,
        'encoder_num_heads': 16,
        'predictor_embed_dim': 384,
        'predictor_depth': 6,
        'predictor_num_heads': 12,
        'c': 8,
        'pos_type': 'sincos',
        'mask_scale': (0, 0),
        'mask_type': 'multiblock' if model_name == 'ejepa_multiblock' else 'random',
        'leads': leads
    }
    encoder = Time_jepa(**params).encoder
    ckpt = torch.load(ckpt_dir)
    encoder.load_state_dict(ckpt['encoder'])
    embed_dim = 768


    return encoder, embed_dim
