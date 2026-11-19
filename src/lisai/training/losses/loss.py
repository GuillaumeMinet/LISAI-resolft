"""
Custom loss functions
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger = logging.getLogger(__name__)


def get_loss_function(name:str,**kwargs):
    """
    Utils function returning instantiation of loss function defined 
    by arg:`name` and parameters specific to the loss (in kwargs).

    """
    assert isinstance(name,str)

    if name in ["MSE","mse","l2","L2"]:
        return torch.nn.MSELoss()
    
    elif name in ["MAE","mae","l1","L1"]:
        return torch.nn.L1Loss()

    elif name == "MSE_upsampling":
        loss_prm = kwargs.get("MSE_upsampling_prm",None)
        assert loss_prm is not None, "MSE upsampling parameters not found"
        
        p = loss_prm.get("upsampling_factor",None)
        alpha = loss_prm.get("alpha",None)
        assert p is not None, "MSE upsampling instation needs attr:`upsampling_factor`"
        if alpha is None:
            logger.warning("MSE upsampling alpha factor not specified, so using default value 0.5")
            alpha = 0.5
        
        return MSE_upsampling(p,alpha)

    elif name in ["CharEdge_loss","CharEdge"]:
        loss_prm = kwargs.get("CharEdge_loss_prm",None)
        if loss_prm is not None:
            alpha = loss_prm.get("alpha",None)
            if alpha is None:
                logger.warning("CharEdge_Loss alpha factor not specified, so using default value 0.05")
                alpha = 0.05
        else:
            logger.warning("CharEdge_Loss alpha factor not specified, so using default value 0.05")
            alpha = 0.05

        return CharEdge_loss(alpha)

    else:
        raise ValueError (f"Loss function {name} unknown.")    



### MSE upsampling ###
class MSE_upsampling(Module):
    """
    Custom MSE loss function applied to upsampled image to avoid ugly upsampling artefacts:
        mse_upsampling = (1-alpha)*normal mse + alpha*CustomMSE"

    WARNING: For now, implemented only for an upsampling factor of 2.

    Parameters:
        p = int., upsampling factor. 
        alpha = float between 0 and 0.9999 - proportion of the "custom" part relative to normal MSE (alpha = 0, normal MSE)

    """

    def __init__(self,p,alpha=0.5):
        super().__init__()
        self.p = p
        self.alpha = alpha        
        logger.info(f"Using MSE upsampling, with alpha = {alpha}")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss(input,target,self.p,self.alpha)


    def loss(self,pred,gt,p=2,alpha=0.5):

        assert alpha >= 0 and alpha < 1, "Custom MSEupsampling: alpha should be a number between 0 (included) and 1 (excluded)"
        assert pred.size(dim=-1) % p == 0, "Can't compute custom MSE-upsampling if images sizes are not multiples of upsampling factor "
        assert p == 2, "Custom MSEupsampling implemented only for an upsampling factor of 2"

        MSE = nn.MSELoss()

        # calculate normal mse
        mse_1 = MSE(pred, gt)

        # sub predictions
        subpred_0 = pred[...,::p,::p]
        subpred_1 = pred[...,1::p,::p]
        subpred_2 = pred[...,::p,1::p]
        subpred_3 = pred[...,1::p,1::p]

        # sub mse on each subpred and average the 3
        subMSE_01 = MSE(subpred_0, subpred_1)
        subMSE_02 = MSE(subpred_0, subpred_2)
        subMSE_03 = MSE(subpred_0, subpred_3)

        mse_2 = 1/3*(subMSE_01 + subMSE_02 + subMSE_03)
        return  mse_1 + alpha*mse_2






### CharEdge_loss ###
class CharEdge_loss(Module):
    """
    Loss as decribe in "Deep Learning enables fast, gentle STED microscopy": 
    https://doi.org/10.1038/s42003-023-05222-1
    
    """

    def __init__(self,alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss(input,target,self.alpha)


    def ch_loss_2D(self,pred, gt):
        "Charbonnier loss"
        
        norm = torch.norm(pred - gt)
        #print(norm)
        norm = torch.squeeze(norm)
        norm = torch.pow(norm, 2)
        norm = norm / (256 * 256) + 1e-6
        norm = torch.pow(norm, 0.5)
        c_loss = torch.mean(norm)
        return c_loss


    def edge_loss_2D(self,pred, gt):
        "Edge loss"

        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],dtype=torch.float32).to(device)
        kernel = kernel.reshape((1, 1, 3, 3))

        bias = torch.tensor([0], dtype=torch.float32).to(device)

        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

        conv.weight = nn.Parameter(kernel) 
        conv.bias = nn.Parameter(bias)  

        pred = conv(pred)
        gt = conv(gt)

        e_loss = self.ch_loss_2D(pred, gt)
        return e_loss


    def loss(self,prediction, gt,alpha=0.05):

        """ 
        Weighed sum of Charbonnier loss and Edge loss.
        """

        c_loss = self.ch_loss_2D(prediction, gt)
        e_loss = self.edge_loss_2D(prediction, gt)
        loss = c_loss + alpha * e_loss

        return loss




#### testing ####

if __name__ == "__main__":
    a=np.zeros((12,12),dtype=np.float32)
    b=np.ones((12,12),dtype=np.float32)
    a = torch.as_tensor(a, device=device)[None, None]
    b = torch.as_tensor(b, device=device)[None, None]

    loss = MSE_upsampling(p=2,alpha=0.2)

    print(loss(a,b)) 