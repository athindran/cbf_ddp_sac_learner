import torch
import torch.nn as nn

class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, ctx_shape, *args):
        super().__init__()
        assert len(obs_shape) == 1
        self.feature_dim = feature_dim

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass

class GRUEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, ctx_shape, *args):
        super().__init__()
        assert len(obs_shape) == 1
        self.feature_dim = feature_dim
        self.gru = nn.GRU(ctx_shape[0], self.feature_dim, num_layers, True, True)

    def forward(self, obs, ctxobs, detach=False):
        #print(ctxobs.shape)
        ctxobs = ctxobs.transpose(1,2)
        outputs, hidden = self.gru(ctxobs)
        
        #print(obs.shape,hidden.shape)
        augobs = torch.cat((obs, hidden[-1,:,:]), dim=-1)
        if detach:
            augobs.detach()
        #print(augobs.shape)
        return augobs
    
    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        self.gru = source.gru
    
    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'identity': IdentityEncoder, 'sequence':GRUEncoder}


def make_encoder(
    encoder_type, obs_shape, ctx_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, ctx_shape, output_logits
    )
