import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.filter_utils import EdgeUtils, Squeeze

def get_temperature(epoch):
    temperature = 30. - (2.9*epoch) if epoch < 10 else 1.
    return temperature

def get_pooling():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

class FeedForward(nn.Module):
    def __init__(self, dims, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(input_dim, dims, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(dims),
            nn.ReLU()
        )
    def forward(self, x, epochs_num=0, std_x=0):
        return self.net(x)

class DynamicFeedForward(nn.Module):
    def __init__(self, dims, input_dim, num_weights):
        super().__init__()
        self.dynamic_conv = DynamicConv(input_dim=input_dim, dims=dims, num_weights=num_weights)
        self.batch_norm = nn.BatchNorm3d(dims)
        self.relu = nn.ReLU()
    def forward(self, x, epochs_num):
        x = self.dynamic_conv(x, epochs_num)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class ShapeDynamicFeedForward(nn.Module):
    def __init__(self, dims, input_dim, num_weights):
        super().__init__()
        self.dynamic_conv = ShapeDynamicConv(input_dim=input_dim, dims=dims, num_weights=num_weights)
        self.batch_norm = nn.BatchNorm3d(dims)
        self.relu = nn.ReLU()
    def forward(self, x, epochs_num):
        x = self.dynamic_conv(x, epochs_num)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class ShapeDynamicFeedForwardSecond(nn.Module):
    def __init__(self, dims, input_dim, num_weights):
        super().__init__()
        self.dynamic_conv = StdDynamicConv(input_dim=input_dim, dims=dims, num_weights=num_weights)
        self.batch_norm = nn.BatchNorm3d(dims)
        self.relu = nn.ReLU()
    def forward(self, x, epochs_num, std_x):
        x = self.dynamic_conv(x, epochs_num, std_x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class DynamicConv(nn.Module):
    def __init__(self, input_dim, dims, num_weights):
        super().__init__()
        # normal version part
        self.weights = nn.Parameter(nn.init.kaiming_normal_(torch.empty((num_weights, dims, input_dim, 3, 3, 3), device="cuda")))
        self.biases = nn.Parameter(nn.init.normal_(torch.empty((num_weights, dims), device="cuda")))

        self.register_parameter("dynamic_weight", self.weights)
        self.register_parameter("dynamic_bias", self.biases)
        # normal version part

        # weights list version part
        # self.weights = [ nn.Parameter(nn.init.kaiming_normal_(torch.empty((dims, input_dim, 3, 3, 3), device="cuda"))) for i in range(num_weights)]
        # self.biases = [nn.Parameter(nn.init.normal_(torch.empty((dims), device="cuda"))) for i in range(num_weights)]

        # for cnt, weight in enumerate(self.weights):
        #     self.register_parameter(f"{cnt}th_dynamic_weight", weight)
        

        # for cnt, bias in enumerate(self.biases):
        #     self.register_parameter(f"{cnt}th_dynamic_bias", bias)
        # list version part

        self.gmp = nn.AdaptiveMaxPool3d(1)
        self.sqeeze_expand = nn.Sequential(nn.Linear(input_dim, input_dim//4+1),
                                           nn.ReLU(),
                                           nn.Linear(input_dim//4+1, num_weights),
                                           )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, epochs_num):
        c, bc, t, h, w = self.weights[0].size()
        n, _, i_t, i_h, i_w = x.size()

        phi_x = self.sqeeze_expand(self.gmp(x).sum(dim=[2,3,4]))
        tau = get_temperature(epoch=epochs_num)
        phi_x = self.softmax(phi_x/tau)

        # list version part
        # dynamic_weight = []
        # dynamic_bias = []
        # for d in range(n):
        #     dynamic_weight += [sum(phi_x[d][g] * self.weights[g] for g in range(len(self.weights)))]
        #     dynamic_bias += [sum(phi_x[d][g] * self.biases[g] for g in range(len(self.biases)))]
        # dynamic_weight = torch.stack(dynamic_weight)
        # dynamic_bias = torch.stack(dynamic_bias)
        # weights = dynamic_weight.view(-1, bc, t, h, w)
        # list version part

        # normal version part
        #(b, w_num) (w_num, dims, input_dim, t, kh, kw) -> (b, dims, input_dim, t, kh, kw)
        dynamic_weight = torch.einsum("bn,nijklm->bijklm", phi_x, self.weights)
        weights = dynamic_weight.view(-1, bc, t, h, w) # (b x dims, input_dim, t, kh, kw)
        #(b, w_num) (w_num, dims)
        dynamic_bias = torch.einsum("bn,ni->bi", phi_x, self.biases)
        # normal version part

        x = x.view(1, -1, i_t, i_h, i_w)
        x = F.conv3d(x, weights, stride=1, padding=1, groups=n)
        x = x.view(n, c, i_t, i_h, i_w) + dynamic_bias[:,:,None,None,None]    
        return x

class ShapeDynamicConv(nn.Module):
    def __init__(self, input_dim, dims, num_weights):
        super().__init__()
        # normal version part
        self.weights = nn.Parameter(nn.init.kaiming_normal_(torch.empty((num_weights, dims, input_dim, 3, 3, 3), device="cuda")))
        self.biases = nn.Parameter(nn.init.normal_(torch.empty((num_weights, dims), device="cuda")))

        self.register_parameter("dynamic_weight", self.weights)
        self.register_parameter("dynamic_bias", self.biases)
        # normal version part

        # weights list version
        # self.weights = [ nn.Parameter(nn.init.kaiming_normal_(torch.empty((dims, input_dim, 3, 3, 3), device="cuda"))) for i in range(num_weights)]
        # self.biases = [nn.Parameter(nn.init.normal_(torch.empty((dims), device="cuda"))) for i in range(num_weights)]

        # for cnt, weight in enumerate(self.weights):
        #     self.register_parameter(f"{cnt}th_dynamic_weight", weight)
        

        # for cnt, bias in enumerate(self.biases):
        #     self.register_parameter(f"{cnt}th_dynamic_bias", bias)
        # list version part

        self.gmp = nn.AdaptiveMaxPool3d(1)
        self.sqeeze_expand = nn.Sequential(nn.Conv2d(input_dim, 16, 5),
                                           nn.ReLU(),
                                           nn.Conv2d(16, 32, 5),
                                           nn.ReLU(),
                                           nn.AdaptiveMaxPool2d(1),
                                           Squeeze(),
                                           nn.Linear(32, num_weights),
                                           )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, epochs_num):
        c, bc, t, h, w = self.weights[0].size()
        n, _, i_t, i_h, i_w = x.size()

        std_x = x.std(axis=(2))

        phi_x = self.sqeeze_expand(std_x)
        tau = get_temperature(epoch=epochs_num)
        phi_x = self.softmax(phi_x/tau)

        # list version part
        # dynamic_weight = []
        # dynamic_bias = []
        # for d in range(n):
        #     dynamic_weight += [sum(phi_x[d][g] * self.weights[g] for g in range(len(self.weights)))]
        #     dynamic_bias += [sum(phi_x[d][g] * self.biases[g] for g in range(len(self.biases)))]
        # dynamic_weight = torch.stack(dynamic_weight)
        # dynamic_bias = torch.stack(dynamic_bias)
        # weights = dynamic_weight.view(-1, bc, t, h, w)
        # list version part

        # normal version part
        #(b, w_num) (w_num, dims, input_dim, t, kh, kw) -> (b, dims, input_dim, t, kh, kw)
        dynamic_weight = torch.einsum("bn,nijklm->bijklm", phi_x, self.weights)
        weights = dynamic_weight.view(-1, bc, t, h, w) # (b x dims, input_dim, t, kh, kw)
        #(b, w_num) (w_num, dims)
        dynamic_bias = torch.einsum("bn,ni->bi", phi_x, self.biases)

        x = x.view(1, -1, i_t, i_h, i_w)
        x = F.conv3d(x, weights, stride=1, padding=1, groups=n)
        x = x.view(n, c, i_t, i_h, i_w) + dynamic_bias[:,:,None,None,None]
        return x

class StdDynamicConv(nn.Module):
    def __init__(self, input_dim, dims, num_weights):
        super().__init__()
        self.weights = nn.Parameter(nn.init.kaiming_normal_(torch.empty((num_weights, dims, input_dim, 3, 3, 3), device="cuda")))
        self.biases = nn.Parameter(nn.init.normal_(torch.empty((num_weights, dims), device="cuda")))

        self.register_parameter("dynamic_weight", self.weights)
        self.register_parameter("dynamic_bias", self.biases)

        self.gmp = nn.AdaptiveMaxPool3d(1)
        self.sqeeze_expand = nn.Sequential(nn.Conv2d(1, 16, 5),
                                           nn.ReLU(),
                                           nn.Conv2d(16, 32, 5),
                                           nn.ReLU(),
                                           nn.AdaptiveMaxPool2d(1),
                                           Squeeze(),
                                           nn.Linear(32, num_weights),
                                           )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, epochs_num, std_x):
        c, bc, t, h, w = self.weights[0].size()
        n, _, i_t, i_h, i_w = x.size()

        phi_x = self.sqeeze_expand(std_x)
        tau = get_temperature(epoch=epochs_num)
        phi_x = self.softmax(phi_x/tau)

        # normal version part
        #(b, w_num) (w_num, dims, input_dim, t, kh, kw) -> (b, dims, input_dim, t, kh, kw)
        dynamic_weight = torch.einsum("bn,nijklm->bijklm", phi_x, self.weights)
        weights = dynamic_weight.view(-1, bc, t, h, w) # (b x dims, input_dim, t, kh, kw)
        #(b, w_num) (w_num, dims)
        dynamic_bias = torch.einsum("bn,ni->bi", phi_x, self.biases)

        x = x.view(1, -1, i_t, i_h, i_w)
        x = F.conv3d(x, weights, stride=1, padding=1, groups=n)
        x = x.view(n, c, i_t, i_h, i_w) + dynamic_bias[:,:,None,None,None]
        return x


class CnnModel(nn.Module):
    def __init__(self, input_dim, frontend_dims, pooling_layer):
        super().__init__()
        self.layers = nn.ModuleList([])
        for layer, dims in enumerate(frontend_dims):
            pooling = get_pooling() if layer+1 in pooling_layer else None
            self.layers.append(nn.ModuleList([
                FeedForward(dims, input_dim),
                pooling
            ]))
            input_dim = dims
    def forward(self, x):
        for ff, pooling in self.layers:
            x = ff(x)
            x = pooling(x) if not isinstance(pooling, type(None)) else x
        return x

class CnnDynamicModel(nn.Module):
    def __init__(self, input_dim, frontend_dims, pooling_layer, num_weights):
        super().__init__()
        self.layers = nn.ModuleList([])
        for layer, dims in enumerate(frontend_dims):
            pooling = get_pooling() if layer+1 in pooling_layer else None
            self.layers.append(nn.ModuleList([
                DynamicFeedForward(dims, input_dim, num_weights),
                pooling
            ]))
            input_dim = dims
    def forward(self, x, epochs_num):
        for ff, pooling in self.layers:
            x = ff(x, epochs_num)
            x = pooling(x) if not isinstance(pooling, type(None)) else x
        return x

class CnnShapeDynamicModel(nn.Module):
    def __init__(self, input_dim, frontend_dims, pooling_layer, num_weights, dynamic_layer):
        super().__init__()
        self.layers = nn.ModuleList([])
        for layer, dims in enumerate(frontend_dims):
            pooling = get_pooling() if layer+1 in pooling_layer else None
            ff = ShapeDynamicFeedForward(dims, input_dim, num_weights) if layer+1 in dynamic_layer else FeedForward(dims, input_dim)
            self.layers.append(nn.ModuleList([
                ff,
                pooling
            ]))
            input_dim = dims
    def forward(self, x, epochs_num):
        for ff, pooling in self.layers:
            x = ff(x, epochs_num)
            x = pooling(x) if not isinstance(pooling, type(None)) else x
        return x

class CnnStdDynamicModel(nn.Module):
    def __init__(self, input_dim, frontend_dims, pooling_layer, num_weights, dynamic_layer):
        super().__init__()
        self.layers = nn.ModuleList([])
        for layer, dims in enumerate(frontend_dims):
            pooling = get_pooling() if layer+1 in pooling_layer else None
            ff = ShapeDynamicFeedForwardSecond(dims, input_dim, num_weights) if layer+1 in dynamic_layer else FeedForward(dims, input_dim)
            self.layers.append(nn.ModuleList([
                ff,
                pooling
            ]))
            input_dim = dims
    def forward(self, x, epochs_num):
        std_x = x.std(axis=2)
        for ff, pooling in self.layers:
            x = ff(x, epochs_num, std_x)
            x = pooling(x) if not isinstance(pooling, type(None)) else x
        return x