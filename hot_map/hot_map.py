import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import os
import time


class Gpmodel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(Gpmodel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def gpr2d(obs1, obs2, target_pos, a = 0.95, b = 0.05):

    epoch = 500

    train_x_obs1 = torch.tensor(obs1[:, 0:2])
    train_y_obs1 = torch.tensor(obs1[:, 2:3].reshape(-1))

    # 设置初始化参数和模型
    likelihood_1 = gpytorch.likelihoods.GaussianLikelihood()
    model_1 = Gpmodel(train_x_obs1, train_y_obs1, likelihood_1)

    model_1.train()
    likelihood_1.train()

    opt_1 = torch.optim.Adam(model_1.parameters(), lr=0.05)
    loss_mll_1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_1, model_1)

    for i in range(epoch):
        opt_1.zero_grad()
        output = model_1(train_x_obs1)
        loss = -(loss_mll_1(output, train_y_obs1))
        loss.backward()
        print('epoch:{:^2}/{:^2} - loss:{:.3f}    lengthscale:{:.3f}   noise:{:.3f}'.format(i + 1, epoch, loss.item(),
                                                                                            model_1.covar_module.base_kernel.lengthscale.item(),
                                                                                            model_1.likelihood.noise.item()))
        opt_1.step()

    model_1.eval()
    likelihood_1.eval()

    with torch.no_grad():
        test_x_obs2 = obs2[:, 0:2]
        test_y_obs2 = obs2[:, 2:3].reshape(-1)
        train_x_obs2 = torch.tensor(obs2[:, 0:2])
        train_y_obs2 = likelihood_1(model_1(train_x_obs2)).mean.numpy()
        merge_y = train_y_obs2 * a + test_y_obs2 * b
        train_x_obs1_np = train_x_obs1.numpy()

        lst_idx = []
        for i in range(len(train_x_obs1)):
            if np.any(np.all(test_x_obs2 == train_x_obs1_np[i], axis=1)):
                lst_idx.append(i)

    train_2_x = np.delete(train_x_obs1, lst_idx, axis=0)
    train_2_x = torch.tensor(np.concatenate((train_2_x, test_x_obs2), axis=0))
    train_2_y = np.delete(train_y_obs1, lst_idx, axis=0)
    train_2_y = torch.tensor(np.concatenate((train_2_y, merge_y), axis=0))

    likelihood_2 = gpytorch.likelihoods.GaussianLikelihood()
    model_2 = Gpmodel(train_2_x, train_2_y, likelihood_2)

    model_2.train()
    likelihood_2.train()

    opt_2 = torch.optim.Adam(model_2.parameters(), lr=0.05)
    loss_mll_2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_2, model_2)

    for i in range(epoch):
        opt_2.zero_grad()
        output = model_2(train_2_x)
        loss = -(loss_mll_2(output, train_2_y))
        loss.backward()
        print('epoch:{:^2}/{:^2} - loss:{:.3f}    lengthscale:{:.3f}   noise:{:.3f}'.format(i + 1, epoch, loss.item(),
                                                                                            model_2.covar_module.base_kernel.lengthscale.item(),
                                                                                            model_2.likelihood.noise.item()))
        opt_2.step()

    model_2.eval()
    likelihood_2.eval()

    target_pos = torch.tensor(target_pos)

    with torch.no_grad():
        result = likelihood_2(model_2(target_pos)).mean.numpy().reshape(-1, 1)

    return np.concatenate((target_pos, result), axis=1)

if __name__ == '__main__':
    time1 = time.time()

    a1 = np.array([[1, 1, 5], [2, 2, 7], [3, 3, 9], [1, 2, 6], [1, 3, 7], [2, 1, 5], [3, 1, 6], [2, 3, 8]], dtype=float)
    a2 = np.array([[1, 1, 4], [3, 2, 6], [4, 3, 9], [1, 4, 7], [3, 3, 8], [2, 4, 8], [3, 4, 9], [2, 3, 7]], dtype=float)
    r = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
                 [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
                 [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]], dtype=float)

    # print(gpr2d(a1, a2, r))


    fig_hot, axe = plt.subplots(figsize=(8, 8))
    img = axe.imshow(gpr2d(a1, a2, r)[:, -1].reshape(6, 6))
    plt.show()

    print('time = {}'.format(time.time() - time1))

