import torch


def compute_cosine_similarity_matrix(feature, threshold=0.5):

    dot_product = torch.mm(feature, torch.transpose(feature, 0, 1))
    norm_squared = torch.sum(feature ** 2, dim=1, keepdim=True)
    denominator = torch.sqrt(torch.mm(norm_squared, norm_squared.transpose(0, 1)))
    denominator[denominator == 0] = 1
    dot_product[dot_product == 0] = -1
    cosine_similarity = dot_product / denominator
    binarized_tensor = (cosine_similarity > threshold).int()
    return binarized_tensor


def DFT(x):
    mean = x.mean(dim = -1)
    std = x.std(dim = -1)
    norm_x = (x - mean.unsqueeze(dim=1)) / std.unsqueeze(dim=1)
    fft_whole = torch.fft.fft(norm_x, dim=-1, norm='forward', n=100)
    real_part = fft_whole.real
    imag_part = fft_whole.imag

    merge_fft = torch.sqrt(real_part ** 2 + imag_part ** 2)
    return merge_fft