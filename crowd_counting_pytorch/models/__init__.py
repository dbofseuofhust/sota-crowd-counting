from .vgg import VGG,vgg19
from .cannet import CANNet
from .sfcn import SFCN
from .sfanet import SFANet
from .asd import ASD
from .scar import SCAR
from .oricannet import OriCANNet
from .c3f import *
from .res50_fpn import Res50_FPN
from .res101_fpn import Res101_FPN

def get_models(model_name,pretrained=True):

    model_dict = {
        'bayesian': vgg19,
        'cannet': CANNet,
        'sfcn': SFCN,
        'sfanet': SFANet,
        'asd': ASD,
        'scar': SCAR,
        'oricannet': OriCANNet,
        'cf3_alexnet': AlexNet,
        'c3f_csrnet': CSRNet,
        'c3f_mcnn': MCNN,
        'c3f_res50': Res50,
        'c3f_res101': Res101,
        'c3f_res101_sfcn': Res101_SFCN,
        'c3f_sanet': SANet,
        'res50_fpn': Res50_FPN,
        'res101_fpn': Res101_FPN
    }

    return model_dict[model_name]