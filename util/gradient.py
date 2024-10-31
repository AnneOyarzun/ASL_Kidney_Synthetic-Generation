import torch

# From Dominiks Cx, axis):
def gradient(x, axis):
    if axis == 0:
        g = torch.sub(x[0,0,1:,:],x[0,0,:-1,:])
    elif axis == 1:
        g = torch.sub(x[0,0,:,1:],x[0,0,:,:-1])
    return torch.abs(g)