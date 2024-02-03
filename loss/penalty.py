from torch.autograd import grad
import torch

def gradient_penalty(x, y, f):
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(x.device)
    z = x + alpha * (y - x)
    z.requires_grad = True
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).to(z.device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    return gp