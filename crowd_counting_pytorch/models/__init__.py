from .vgg import VGG,vgg19
from .cannet import CANNet
from .sfcn import SFCN
from .sfanet import SFANet
from .asd import ASD
from .scar import SCAR
from .oricannet import OriCANNet

def get_models(model_name,pretrained=True):

    model_dict = {
        'bayesian': vgg19,
        'cannet': CANNet,
        'sfcn': SFCN,
        'sfanet': SFANet,
        'asd': ASD,
        'scar': SCAR,
        'oricannet': OriCANNet,
    }

    return model_dict[model_name]