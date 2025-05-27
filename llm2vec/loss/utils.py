from .HardNegativeNLLLoss import HardNegativeNLLLoss,DEBATERHardNegativeNLLLoss,MultiViewsHardNegativeNLLLoss


def load_loss(loss_class, *args, **kwargs):
    print(loss_class)
    print(type(loss_class))
    print(loss_class == "NORMAL")
    if loss_class == "NORMAL":
        loss_cls = HardNegativeNLLLoss
    elif loss_class == "DEBATER":
        loss_cls = DEBATERHardNegativeNLLLoss
    elif loss_class == "MULTI-VIEW":
        loss_cls = MultiViewsHardNegativeNLLLoss
    else:
        raise ValueError(f"Unknown loss class {loss_class}")
    return loss_cls(*args, **kwargs)
