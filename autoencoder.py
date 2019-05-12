import torch
import torch.nn as nn
from encoder_decoder_factory import Encoder, Decoder

def wct(alpha, cf, sf, s1f=None, beta=None):
    # content image whitening
    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    cfv = cf.view(c_channels, -1)  # c x (h x w)

    c_mean = torch.mean(cfv, 1) # perform mean for each row
    c_mean = c_mean.unsqueeze(1).expand_as(cfv) # add dim and replicate mean on rows
    cfv = cfv - c_mean # subtract mean element-wise

    c_covm = torch.mm(cfv, cfv.t()).div((c_width * c_height) - 1)  # construct covariance matrix
    c_u, c_e, c_v = torch.svd(c_covm, some=False) # singular value decomposition

    k_c = c_channels
    for i in range(c_channels):
        if c_e[i] < 0.00001:
            k_c = i
            break
    c_d = (c_e[0:k_c]).pow(-0.5)

    w_step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    w_step2 = torch.mm(w_step1, (c_v[:, 0:k_c].t()))
    whitened = torch.mm(w_step2, cfv)

    # style image coloring
    sf = sf.double()
    _, s_width, s_heigth = sf.size(0), sf.size(1), sf.size(2)
    sfv = sf.view(c_channels, -1)

    s_mean = torch.mean(sfv, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(sfv)
    sfv = sfv - s_mean

    s_covm = torch.mm(sfv, sfv.t()).div((s_width * s_heigth) - 1)
    s_u, s_e, s_v = torch.svd(s_covm, some=False)

    s_k = c_channels # same number of channels ad content features
    for i in range(c_channels):
        if s_e[i] < 0.00001:
            s_k = i
            break
    s_d = (s_e[0:s_k]).pow(0.5)

    c_step1 = torch.mm(s_v[:, 0:s_k], torch.diag(s_d))
    c_step2 = torch.mm(c_step1, s_v[:, 0:s_k].t())
    colored = torch.mm(c_step2, whitened)

    cs0_features = colored + s_mean.resize_as_(colored)
    cs0_features = cs0_features.view_as(cf)
    target_features = cs0_features
    ccsf = alpha * target_features + (1.0 - alpha) * cf
    return ccsf.float().unsqueeze(0)



def stylize(level, content, style0, encoders, decoders, alpha, svd_device, cnn_device):
    with torch.no_grad():

        cf = encoders[level](content).data.to(device=svd_device).squeeze(0)
        s0f = encoders[level](style0).data.to(device=svd_device).squeeze(0)
        csf = wct(alpha, cf, s0f).to(device=cnn_device)

        return decoders[level](csf)


class MultiLevelWCT(nn.Module):

    def __init__(self, args):
        super(MultiLevelWCT, self).__init__()
        self.svd_device = torch.device('cpu')
        self.cnn_device = args.device
        self.alpha = args.alpha
        self.e1, self.e2, self.e3, self.e4, self.e5 = Encoder(1), Encoder(2), Encoder(3), Encoder(4), Encoder(5)
        self.encoders = [self.e5, self.e4, self.e3, self.e2, self.e1]
        self.d1, self.d2, self.d3, self.d4, self.d5 = Decoder(1), Decoder(2), Decoder(3), Decoder(4), Decoder(5)
        self.decoders = [self.d5, self.d4, self.d3, self.d2, self.d1]
    def forward(self, content_img, style_img):
        for i in range(len(self.encoders)):
                content_img = stylize(i, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,self.cnn_device)
        return content_img

