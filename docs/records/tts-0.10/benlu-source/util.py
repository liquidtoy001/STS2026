import torch
import torch.nn.functional as F

def sequence_mask(lengths, max_length=None):
    if max_length is None:
        max_length = int(lengths.max().item())
    return torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]

def slice_segments(x, ids_str=None, x_lengths=None, segment_size = 4):
    b, d, t = x.size()
    
    if ids_str is None:
        if x_lengths is None:
            x_lengths = t
        ids_str_max = x_lengths - segment_size + 1
        ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(b):
        ret[i] = x[i, :, ids_str[i]:ids_str[i] + segment_size]
    return ret, ids_str

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device
    
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2,3) * mask
    return path