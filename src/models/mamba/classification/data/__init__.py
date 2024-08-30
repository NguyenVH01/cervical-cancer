from .build import build_loader as _build_loader, build_custom_dataset
from .data_simmim_pt import build_loader_simmim
from .data_simmim_ft import build_loader_finetune

def build_loader(config, simmim=False, is_pretrain=False):
    return build_custom_dataset(config)
    # if not simmim:
    #     return _build_loader(config)
    # if is_pretrain:
    #     return build_loader_simmim(config)
    # else:
    #     return build_loader_finetune(config)
