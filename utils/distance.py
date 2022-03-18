import torch
import torch.nn.functional as F


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    # KLDivLoss = F.kl_div(reduction='sum')
    if get_softmax:
        p_output = F.softmax(p_output,dim=-1)
        q_output = F.softmax(q_output,dim=-1)
    log_mean_output = ((p_output + q_output )/2).log()
    return (F.kl_div(log_mean_output, p_output,reduction='sum') + F.kl_div(log_mean_output, q_output,reduction='sum'))/2
