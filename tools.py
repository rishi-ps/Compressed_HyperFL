import copy
import torch

def average_weights_weighted(w, avg_weight):
    """
    Compute weighted average of model state_dicts.
    Args:
        w: list of state_dicts from clients
        avg_weight: tensor of client weights (number of samples)
    Returns:
        w_avg: aggregated state_dict
    """
    w_avg = copy.deepcopy(w[0])
    # Use detach().clone() to avoid UserWarning
    weight = avg_weight.detach().clone()
    agg_w = weight / weight.sum(dim=0)

    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key]).float()
        for i in range(len(w)):
            w_avg[key] += agg_w[i] * w[i][key].float()

    return w_avg


def get_parameter_values(model):
    """
    Flatten model parameters into a single 1D tensor
    """
    parameter = torch.cat([param.data.reshape(-1) for param in model.parameters()]).detach().clone()
    return parameter


def gaussian_noise(data_shape, clip, sigma, device=None):
    """
    Generate Gaussian noise tensor
    """
    return torch.normal(0, sigma * clip, size=data_shape).to(device)
