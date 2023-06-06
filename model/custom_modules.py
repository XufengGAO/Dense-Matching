r"""Implementation of Dynamic Layer Gating (DLG)"""
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

class DynamicFeatureSelection(nn.Module):
    def __init__(self, in_channels, N, neck):
        super(DynamicFeatureSelection, self).__init__()
        self.in_channels = in_channels
        self.C = sum(in_channels)
        self.w = nn.Parameter(torch.zeros((N, neck.D))) # N x D
        self.use_bias = neck.use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1,neck.D,1,1,1)) #bdchw
        self.init_weights(neck.init_type)

        self.use_mp = neck.use_mp
        if self.use_mp:
            K = []
            start = 0
            for ch in self.in_channels:
                i = torch.arange(start, start+ch).repeat(2,1)
                v = torch.ones(ch)
                k = torch.sparse.FloatTensor(i, v, torch.Size([self.C, self.C]))
                K.append(k)
                start += ch
            self.K = torch.stack(K, dim=0).cuda()

        self.use_relu = neck.use_relu

    def init_weights(self, init_type):
        if init_type == 'kaiming_norm':
            nn.init.kaiming_normal_(self.w.data)
        elif init_type == 'xavier_norm':
            nn.init.xavier_normal_(self.w.data)
        else:
            nn.init.uniform_(self.w.data)


    def forward(self, feat):

        # matric product, feat shape in [B, C, HW]
        if self.use_mp:
            feat = torch.cat(feat, dim=1)  # list of [B, Ci, H, W] -> [B, C, H, W]
            KF = torch.einsum('ncc,bchw->bnchw', self.K.to_dense(), feat).to_sparse()
            if self.use_relu:
                return F.relu(torch.einsum('dn, bnchw->bdchw', self.w.T, KF.to_dense()), inplace=True)
            else:
                return torch.einsum('dn, bnchw->bdchw', self.w.T, KF.to_dense())
        else:
            # for-loop, feat is a list
            result = []
            for i, f in enumerate(feat):
                if self.use_bias:
                    result.append(torch.einsum('d, bchw->bdchw', self.w.T[:,i], f)+self.bias)
                else:
                    result.append(torch.einsum('d, bchw->bdchw', self.w.T[:,i], f))

            if self.use_relu:
                return F.relu(torch.cat(result, dim=2), inplace=True)
            else:
                return torch.cat(result, dim=2)

class GradNorm(nn.Module):
    def __init__(self, num_of_task, alpha=1.5):
        super(GradNorm, self).__init__()
        self.num_of_task = num_of_task
        self.alpha = alpha
        self.w = nn.Parameter(torch.ones(num_of_task, dtype=torch.float))
        self.l1_loss = nn.L1Loss()
        self.L_0 = None

    # standard forward pass
    def forward(self, L_t: torch.Tensor):
        # initialize the initial loss `Li_0`
        if self.L_0 is None:
            self.L_0 = L_t.detach() # detach
        # compute the weighted loss w_i(t) * L_i(t)
        self.L_t = L_t
        self.wL_t = L_t * self.w
        # the reduced weighted loss
        self.total_loss = self.wL_t.sum()
        return self.total_loss

    # additional forward & backward pass
    def additional_forward_and_backward(self, grad_norm_weights: nn.Module, optimizer: optim.Optimizer):
        # do `optimizer.zero_grad()` outside
        self.total_loss.backward(retain_graph=True)
        # in standard backward pass, `w` does not require grad
        self.w.grad.data = self.w.grad.data * 0.0

        self.GW_t = []
        for i in range(self.num_of_task):
            # get the gradient of this task loss with respect to the shared parameters
            GiW_t = torch.autograd.grad(
                self.L_t[i], grad_norm_weights.parameters(),
                    retain_graph=True, create_graph=False)
            
            # GiW_t is tuple
            # compute the norm
            self.GW_t.append(torch.norm(GiW_t[0] * self.w[i]))

        self.GW_t = torch.stack(self.GW_t) # do not detatch
        self.bar_GW_t = self.GW_t.detach().mean()
        self.tilde_L_t = (self.L_t / self.L_0).detach()
        self.r_t = self.tilde_L_t / self.tilde_L_t.mean()
        grad_loss = self.l1_loss(self.GW_t, self.bar_GW_t * (self.r_t ** self.alpha))
        self.w.grad = torch.autograd.grad(grad_loss, self.w)[0]
        optimizer.step()

        self.GW_ti, self.bar_GW_t, self.tilde_L_t, self.r_t, self.L_t, self.wL_t = None, None, None, None, None, None
        # re-norm
        self.w.data = self.w.data / self.w.data.sum() * self.num_of_task