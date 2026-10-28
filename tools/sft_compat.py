"""Explicit source-backed numerical fixes for the original JAX checkpoint path."""
from types import MethodType


def siglip_mlp_forward(self,x):
    import torch.nn.functional as F
    x=self.fc1(x)
    x=F.gelu(x,approximate='tanh')
    x=self.dropout(x)
    return self.fc2(x)


def apply_original_siglip_activation(model):
    from rlinf.models.embodiment.openpi.modules.siglip import MlpBlock
    names=[]
    for name,module in model.named_modules():
        if name.startswith('img.') and isinstance(module,MlpBlock):
            module.forward=MethodType(siglip_mlp_forward,module);names.append(name)
    assert len(names)==27, names
    return names
