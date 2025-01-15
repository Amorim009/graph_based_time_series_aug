from neuralforecast.models import NHITS, MLP, KAN
from metaforecast.synth import (SeasonalMBB,
                                Jittering,
                                Scaling,
                                MagnitudeWarping,
                                TimeWarping,
                                DBA,
                                TSMixup)
from pytorch_lightning import Trainer

trainer = Trainer(accelerator='cpu')

MODEL = 'KAN'
ACCELERATOR = 'cpu'

MODELS = {
    'NHITS': NHITS,
    'MLP': MLP,
    'KAN': KAN,
}

MODEL_CONFIG = {
    'NHITS': {
        'start_padding_enabled': False,
        'accelerator': ACCELERATOR,
        'scaler_type': 'standard',
        'max_steps': 1000,
    },
    'MLP': {
        'start_padding_enabled': False,
        'accelerator': ACCELERATOR,
        'scaler_type': 'standard',
        'max_steps': 1000,
    },
    'KAN': {
        'accelerator': ACCELERATOR,
        'scaler_type': 'standard',
        'max_steps': 1000,
    },

}

SYNTH_METHODS = {
    'SeasonalMBB': SeasonalMBB,
    'Jittering': Jittering,
    'Scaling': Scaling,
    'TimeWarping': TimeWarping,
    'MagnitudeWarping': MagnitudeWarping,
    'TSMixup': TSMixup,
    'DBA': DBA,
}

SYNTH_METHODS_ARGS = {
    'SeasonalMBB': ['seas_period'],
    'Jittering': [],
    'Scaling': [],
    'MagnitudeWarping': [],
    'TimeWarping': [],
    'DBA': ['max_n_uids'],
    'TSMixup': ['max_n_uids', 'max_len', 'min_len']
}
