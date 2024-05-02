from .swin_transformer.swin_transformer import swin_base_patch4_window7_224
import torch


def load_model_with_weight(model_name: str):
    if model_name == 'swin_base':
        model = swin_base_patch4_window7_224(convert_weights=False, semantic_weight=1.0)
        model.init_weights('weights/swin_base.pth')
    else:
        model = None

    return model