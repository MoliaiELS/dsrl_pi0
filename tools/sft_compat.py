"""Explicit source-backed numerical fixes for the original JAX checkpoint path."""
from types import MethodType
import math


def original_gelu(x):
    """JAX nn.gelu's operations, including its bfloat16 intermediate rounding.

    F.gelu(tanh) evaluates internal expressions in float32 for bf16 inputs;
    JAX's typed elementary operations round between operations instead.
    """
    import torch
    scalar=lambda v:x.new_tensor(v)
    cdf=scalar(.5)*(scalar(1.)+torch.tanh(scalar(math.sqrt(2/math.pi))*(x+scalar(.044715)*(x*x*x))))
    return x*cdf


def original_gelu_glu(gate,value):
    return original_gelu(gate)*value


def siglip_mlp_forward(self,x):
    x=self.fc1(x)
    x=original_gelu(x)
    x=self.dropout(x)
    return self.fc2(x)


def apply_original_siglip_activation(model):
    from rlinf.models.embodiment.openpi.modules import gemma
    from rlinf.models.embodiment.openpi.modules.siglip import MlpBlock
    names=[]
    for name,module in model.named_modules():
        if name.startswith('img.') and isinstance(module,MlpBlock):
            module.forward=MethodType(siglip_mlp_forward,module);names.append(name)
    assert len(names)==27, names
    # Original gemma.py MLP also calls Flax nn.gelu(approximate=True).
    # Only this process's function binding changes; upstream files stay pinned.
    gemma.gelu_glu=original_gelu_glu
    return names
