import torch
import numpy as np

# Varies for different model
# mapping the name of conv to its corresponding bn
def mapping(conv_name):
    try:
        return conv_name[:-1] + str(int(conv_name[-1])+1)
    except ValueError:
        return 'None'


class InfoStruct(object):

    def __init__(self, module, pre_f_cls, f_cls, b_cls):

        # init
        self.module = module
        self.pre_f_cls = pre_f_cls
        self.f_cls = f_cls
        self.b_cls = b_cls

        # forward statistic
        self.forward_mean = None
        self.variance = None
        self.forward_cov = None
        self.channel_num = None
        self.zero_variance_masked_zero = None
        self.zero_variance_masked_one = None
        self.alpha = None  # variance after de-correlation
        self.stack_op_for_weight = None

        # backward statistic
        self.grad_mean = None
        self.grad_cov = None
        self.adjust_matrix = None

        # score
        self.score = None
        self.sorted_index = None

        # parameters form model
        self.weight = None

        # assigned before pruning
        self.bn_module = None

    def compute_statistic_and_fetch_weight(self):
        # compute forward covariance
        self.forward_mean = self.f_cls.sum_mean / self.f_cls.counter
        self.forward_cov = (self.f_cls.sum_covariance / self.f_cls.counter) - \
            torch.mm(self.forward_mean.view(-1, 1), self.forward_mean.view(1, -1))

        self.channel_num = list(self.forward_cov.shape)[0]

        # equal 0 where variance of an activate is 0
        self.variance = torch.diag(self.forward_cov)
        self.zero_variance_masked_zero = torch.sign(self.variance)

        # where 0 var compensate 1
        self.zero_variance_masked_one = torch.nn.Parameter(torch.ones(self.channel_num, dtype=torch.double)).cuda() - \
            self.zero_variance_masked_zero
        repaired_forward_cov = self.forward_cov + torch.diag(self.zero_variance_masked_one)

        f_cov_inverse = repaired_forward_cov.inverse().to(torch.float)
        repaired_alpha = torch.reciprocal(torch.diag(f_cov_inverse))
        self.alpha = repaired_alpha *  self.zero_variance_masked_zero.to(torch.float)

        self.stack_op_for_weight = (f_cov_inverse.t() * repaired_alpha.view(1, -1)).t()

        self.weight = self.module.weight.detach()

        self.grad_mean = self.b_cls.sum_mean / self.b_cls.counter
        self.grad_cov = (self.b_cls.sum_covariance / self.b_cls.counter) - \
            torch.mm(self.grad_mean.view(-1, 1), self.grad_mean.view(1, -1))
        eig_value, eig_vec = torch.eig(self.grad_cov, eigenvectors=True)

        self.adjust_matrix = torch.mm(torch.diag(torch.sqrt(eig_value[:, 0])), eig_vec.t()).to(torch.float)
        # print('M: ', adjust_matrix.shape)

        adjusted_weight = torch.norm(torch.mm(self.adjust_matrix, torch.squeeze(self.weight)), dim=0)

        self.score = adjusted_weight * self.alpha
        self.sorted_index = torch.argsort(self.score)
        print(torch.sort(self.score)[0])

    def clear_zero_variance(self):

        # according to zero variance mask, remove all the channels with 0 variance,
        # this function first update [masks] in pre_forward_hook,
        # then update parameters in [bn module] or biases in the last layer

        self.pre_f_cls.update_mask(self.zero_variance_masked_zero.to(torch.float))

        print('remove activate: ', torch.sum(self.zero_variance_masked_one))

        used_mean = self.forward_mean.to(torch.float) * self.zero_variance_masked_one.to(torch.float)
        repair_base = torch.squeeze(torch.mm(torch.squeeze(self.weight), used_mean.view(-1, 1)))

        if self.bn_module is None:
            print('Modify biases in', self.module)
            self.module.bias.data -= repair_base
        else:
            self.bn_module.running_mean.data -= repair_base


def compute_statistic_and_update(samples, sum_mean, sum_covar, counter) -> None:
    samples = samples.to(torch.half).to(torch.double)
    samples_num = list(samples.shape)[0]
    counter += samples_num
    sum_mean += torch.sum(samples, dim=0)
    sum_covar += torch.mm(samples.permute(1, 0), samples)


class ForwardStatisticHook(object):

    def __init__(self, name=None, dim=4):
        self.name = name
        self.dim = dim
        self.sum_mean = None
        self.sum_covariance = None
        self.counter = None

    def __call__(self, module, inputs, output) -> None:
        with torch.no_grad():
            channel_num = list(inputs[0].shape)[1]
            if self.sum_mean is None or self.sum_covariance is None:
                self.sum_mean = torch.nn.Parameter(torch.zeros(channel_num).to(torch.double),
                                                   requires_grad=False).cuda()
                self.sum_covariance = \
                    torch.nn.Parameter(torch.zeros(channel_num, channel_num).to(torch.double),
                                       requires_grad=False).cuda()
                self.counter = torch.nn.Parameter(torch.zeros(1).to(torch.double), requires_grad=False).cuda()
            # from [N,C,W,H] to [N*W*H,C]
            if self.dim == 4:
                samples = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            elif self.dim == 2:
                samples = inputs[0]
            compute_statistic_and_update(samples, self.sum_mean, self.sum_covariance, self.counter)


class BackwardStatisticHook(object):

    def __init__(self, name=None, dim=4):
        self.name = name
        self.dim = dim
        self.sum_covariance = None
        self.sum_mean = None
        self.counter = None

    def __call__(self, module, grad_input, grad_output) -> None:
        with torch.no_grad():
            channel_num = list(grad_output[0].shape)[1]
            if self.sum_covariance is None:
                self.sum_mean = torch.nn.Parameter(torch.zeros(channel_num).to(torch.double),
                                                   requires_grad=False).cuda()
                self.sum_covariance = \
                    torch.nn.Parameter(torch.zeros(channel_num, channel_num).to(torch.double),
                                       requires_grad=False).cuda()
                self.counter = torch.nn.Parameter(torch.zeros(1).to(torch.double), requires_grad=False).cuda()
            if self.dim == 4:
                samples = grad_output[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            elif self.dim == 2:
                samples = grad_output[0]
            compute_statistic_and_update(samples, self.sum_mean, self.sum_covariance, self.counter)


class PreForwardHook(object):

    def __init__(self, name, dim=4):
        self.name = name
        self.dim=dim
        self.mask = None
        self.base = None

    def __call__(self, module, inputs):
        channel_num = list(inputs[0].shape)[1]
        if self.mask is None:
            self.mask = torch.nn.Parameter(torch.ones(channel_num), requires_grad=False).cuda()
        if self.dim == 4:
            modified = torch.mul(inputs[0].permute([0, 2, 3, 1]), self.mask)
            return tuple(modified.permute([0, 3, 1, 2]), )
        elif self.dim == 2:
            return tuple(torch.mul(inputs[0], self.mask))

    def update_mask(self, new_mask):
        self.mask.data = new_mask


class StatisticManager(object):

    def __init__(self):

        self.name_to_statistic = {}
        self.bn_name = {}

    def __call__(self, model):

        for name, sub_module in model.named_modules():

            if isinstance(sub_module, torch.nn.Conv2d):
                if sub_module.kernel_size[0] == 1:
                    pre_hook_cls = PreForwardHook(name)
                    hook_cls = ForwardStatisticHook(name)
                    back_hook_cls = BackwardStatisticHook(name)
                    sub_module.register_forward_pre_hook(pre_hook_cls)
                    sub_module.register_forward_hook(hook_cls)
                    sub_module.register_backward_hook(back_hook_cls)
                    self.name_to_statistic[name] = InfoStruct(sub_module, pre_hook_cls, hook_cls, back_hook_cls)
                print('conv', name)

            elif isinstance(sub_module, torch.nn.Linear):
                pre_hook_cls = PreForwardHook(name, dim=2)
                hook_cls = ForwardStatisticHook(name, dim=2)
                back_hook_cls = BackwardStatisticHook(name, dim=2)
                sub_module.register_forward_pre_hook(pre_hook_cls)
                sub_module.register_forward_hook(hook_cls)
                sub_module.register_backward_hook(back_hook_cls)
                self.name_to_statistic[name] = InfoStruct(sub_module, pre_hook_cls, hook_cls, back_hook_cls)
                print('conv', name)

            elif isinstance(sub_module, torch.nn.BatchNorm1d) or isinstance(sub_module, torch.nn.BatchNorm2d):
                self.bn_name[name] = sub_module
                print('bn', name)

    def computer_score(self):

        with torch.no_grad():

            for name in self.name_to_statistic:

                info = self.name_to_statistic[name]

                if mapping(name) in self.bn_name:
                    info.bn_module = self.bn_name[mapping(name)]

                info.compute_statistic_and_fetch_weight()

                info.clear_zero_variance()

    def visualize(self):

        from matplotlib import pyplot as plt
        i = 1
        for name in self.name_to_statistic:
            info = self.name_to_statistic[name]
            forward_mean = info.f_cls.sum_mean / info.f_cls.counter
            forward_cov = (info.f_cls.sum_covariance / info.f_cls.counter) - \
                torch.mm(forward_mean.view(-1, 1), forward_mean.view(1, -1))

            grad_mean = info.b_cls.sum_mean / info.b_cls.counter
            grad_cov = (info.b_cls.sum_covariance / info.b_cls.counter) - \
                torch.mm(grad_mean.view(-1, 1), grad_mean.view(1, -1))
            plt.subplot(10, 15, i)
            plt.imshow(np.array(forward_cov.cpu()), cmap='hot')
            plt.xticks([])
            plt.yticks([])
            i += 1
            plt.subplot(10, 15, i)
            plt.imshow(np.array(grad_cov.cpu()), cmap='hot')
            plt.xticks([])
            plt.yticks([])
            i += 1
            if i > 150:
                break
        plt.show()
