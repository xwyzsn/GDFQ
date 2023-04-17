"""
basic trainer
"""
import time

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
from einops import repeat

__all__ = ["Trainer"]


def exist(val):
    if val is not None:
        return True
    return False


def default(val, d):
    if exist(val):
        return val
    return d() if callable(d) else d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion * planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion * planes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn) -> None:
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SimeseNetwork(nn.Module):
    def __init__(self, in_channels, dim=32*32):
        super(SimeseNetwork, self).__init__()
        # img size = 32

        self.model = nn.Sequential(
            BasicBlock(in_channels, 64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64*16*16, dim),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        output1 = self.model(x1)
        output2 = self.model(x2)
        dis = (output1 - output2) ** 2
        output = self.out(dis)
        return output


class Trainer(object):
    """
    trainer for training network, use SGD
    """

    def __init__(self, model, model_teacher, generator, lr_master_S, lr_master_G,
                 train_loader, test_loader, settings, logger, tensorboard_logger=None,
                 opt_type="SGD", optimizer_state=None, run_count=0):
        """
        init trainer
        """

        self.settings = settings

        self.model = utils.data_parallel(
            model, self.settings.nGPU, self.settings.GPU)
        self.model_teacher = utils.data_parallel(
            model_teacher, self.settings.nGPU, self.settings.GPU)

        self.generator = utils.data_parallel(
            generator, self.settings.nGPU, self.settings.GPU)

        self.simese = utils.data_parallel(
            SimeseNetwork(3, self.settings.nClasses), self.settings.nGPU, self.settings.GPU)

        self.opt_siam = torch.optim.Adam(self.simese.parameters(), lr=self.settings.lr_G,
                                         betas=(self.settings.b1, self.settings.b2))

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tensorboard_logger = tensorboard_logger
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.bce_logits = nn.BCEWithLogitsLoss().cuda()
        self.MSE_loss = nn.MSELoss().cuda()
        self.lr_master_S = lr_master_S
        self.lr_master_G = lr_master_G
        self.opt_type = opt_type
        if opt_type == "SGD":
            self.optimizer_S = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weightDecay,
                nesterov=True,
            )
        elif opt_type == "RMSProp":
            self.optimizer_S = torch.optim.RMSprop(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1.0,
                weight_decay=self.settings.weightDecay,
                momentum=self.settings.momentum,
                alpha=self.settings.momentum
            )
        elif opt_type == "Adam":
            self.optimizer_S = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1e-5,
                weight_decay=self.settings.weightDecay
            )
        else:
            assert False, "invalid type: %d" % opt_type
        if optimizer_state is not None:
            self.optimizer_S.load_state_dict(optimizer_state)\

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
                                            betas=(self.settings.b1, self.settings.b2))

        self.logger = logger
        self.run_count = run_count
        self.scalar_info = {}
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.save_BN_mean = []
        self.save_BN_var = []

        self.fix_G = False

        self.buffer = torch.zeros((self.settings.nClasses,
                                  3, self.settings.img_size, self.settings.img_size)).cuda()
        self.img_batch = None
        self.label = None
        self.output_teacher_batch = None
        self.teacher_feature_extract = []

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        lr_S = self.lr_master_S.get_lr(epoch)
        lr_G = self.lr_master_G.get_lr(epoch)
        # update learning rate of model optimizer
        for param_group in self.optimizer_S.param_groups:
            param_group['lr'] = lr_S

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr_G

    def loss_fn_kd(self, output, labels, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha

        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """

        criterion_d = nn.CrossEntropyLoss().cuda()
        kdloss = nn.KLDivLoss().cuda()

        alpha = self.settings.alpha
        T = self.settings.temperature
        a = F.log_softmax(output / T, dim=1)
        b = F.softmax(teacher_outputs / T, dim=1)
        c = (alpha * T * T)
        d = criterion_d(output, labels)

        KD_loss = kdloss(a, b)*c + d
        return KD_loss

    def forward(self, images, teacher_outputs, labels=None):
        """
        forward propagation
        """
        # forward and backward and optimize
        output, output_1 = self.model(images, True)
        if labels is not None:
            loss = self.loss_fn_kd(output, labels, teacher_outputs)
            return output, loss
        else:
            return output, None

    def backward_G(self, lossG):
        """
        backward propagation
        """
        self.optimizer_G.zero_grad()
        lossG.backward()
        self.optimizer_G.step()

    def backward_S(self, loss_S):
        """
        backward propagation
        """
        self.optimizer_S.zero_grad()
        loss_S.backward()
        self.optimizer_S.step()

    def backward(self, loss):
        """
        backward propagation
        """
        self.optimizer_G.zero_grad()
        self.optimizer_S.zero_grad()
        loss.backward()
        self.optimizer_G.step()
        self.optimizer_S.step()

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def hook_fn_forward_saveBN(self, module, input, output):
        self.save_BN_mean.append(module.running_mean.cpu())
        self.save_BN_var.append(module.running_var.cpu())

    def save_image_center(self, output_teacher, labels, images):
        beta = 0.9
        idx = torch.argmax(output_teacher, dim=1)
        right_idx = torch.nonzero(idx == labels)
        if torch.all(self.buffer[labels[right_idx].squeeze(1), :, :, :] == 0.):
            self.buffer[labels[right_idx].squeeze(
                1), :, :, :] = images[right_idx].squeeze(1)
        else:
            self.buffer[labels[right_idx].squeeze(1), :, :, :] = beta * (self.buffer[labels[right_idx].squeeze(1), :, :, :] +
                                                                         images[right_idx].squeeze(1)) * (1 - beta)

    def train_simese(self, images, labels, teacher_output):

        center = self.buffer[labels, :, :, :].squeeze(1).clone().detach()
        sim_label = (torch.argmax(teacher_output, dim=1) == labels).float()
        sim = self.simese(images, center).squeeze(1)
        loss = F.binary_cross_entropy(sim, sim_label.float())
        self.opt_siam.zero_grad()
        loss.backward()
        self.opt_siam.step()
        return sim, loss

    def sim_predict(self, images, labels):
        center = self.buffer[labels, :, :, :].squeeze(1).clone().detach()
        sim = self.simese(images, center).squeeze(1)
        return sim

    def cal_sim(self, output_teacher, labels, images) -> torch.Tensor:
        idx = torch.argmax(output_teacher, dim=1)
        right_idx = torch.nonzero(idx == labels)
        if right_idx.size(0) == 0:
            return torch.tensor(0.)
        imgs = images[right_idx].squeeze(1)
        center = self.buffer[labels[right_idx].squeeze(1), :, :, :]
        sim = F.cosine_similarity(imgs.view(
            imgs.size(0), 3, -1), center.view(center.size(0), 3, -1), dim=2).sum(dim=1)

        return sim

    def sim_loss(self, sim, threshold=0.5):
        loss = torch.zeros_like(sim).cuda()
        loss = torch.where(
            sim < threshold, torch.tensor(0.).cuda(), -torch.log(1 - sim).cuda())

        return loss.sum()

    def collect_passing_threshold(self, sim, images, labels, output_teacher, threshold=0.5):
        idx = torch.nonzero(sim <= threshold)
        if idx.numel() <= 0:
            return
        argidx = torch.argmax(output_teacher, dim=1)
        right_idx = torch.nonzero(argidx == labels)
        if right_idx.numel() <= 0:
            return
        corr_images = images[right_idx].squeeze(1)
        corr_labels = labels[right_idx].squeeze(1)
        corr_output_teacher = output_teacher[right_idx].squeeze(1)
        passing_images = corr_images[idx].squeeze(1)
        if passing_images.numel() == 0:
            return
        passing_labels = corr_labels[idx].squeeze(1)
        if exist(self.img_batch):
            self.img_batch = torch.cat((self.img_batch, passing_images), dim=0)
            self.label_batch = torch.cat(
                (self.label_batch, passing_labels), dim=0)
            self.output_teacher_batch = torch.cat(
                (self.output_teacher_batch, corr_output_teacher[idx].squeeze(1)), dim=0)
        else:
            self.img_batch = passing_images
            self.label_batch = passing_labels
            self.output_teacher_batch = corr_output_teacher[idx].squeeze(1)

    def register_siamese_hook(self, module, input, output):
        self.teacher_feature_extract.append(output)

    def train(self, epoch):
        """
        training
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()
        fp_acc = utils.AverageMeter()

        iters = 200
        self.update_lr(epoch)

        self.model.eval()
        self.model_teacher.eval()
        self.generator.train()

        start_time = time.time()
        end_time = start_time

        if epoch == 0:
            for m in self.model_teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.register_forward_hook(self.hook_fn_forward)
            # for n, p in self.model_teacher.named_modules():
            #     if n in
        siamese_loss = []
        for i in range(iters):
            start_time = time.time()
            data_time = start_time - end_time
            # while not exist(self.img_batch) or (self.img_batch.size(0) < self.settings.batchSize):

            z = Variable(torch.randn(self.settings.batchSize,
                                     self.settings.latent_dim)).cuda()

            # Get labels ranging from 0 to n_classes for n rows
            labels = Variable(torch.randint(
                0, self.settings.nClasses, (self.settings.batchSize,))).cuda()
            z = z.contiguous()
            labels = labels.contiguous()
            images = self.generator(z, labels)

            self.mean_list.clear()
            self.var_list.clear()
            output_teacher_batch, output_teacher_1 = self.model_teacher(
                images, out_feature=True)

            self.save_image_center(
                output_teacher_batch, labels, images)
            # One hot loss
            loss_one_hot = self.criterion(output_teacher_batch, labels)

            # BN statistic loss
            BNS_loss = torch.zeros(1).cuda()

            for num in range(len(self.mean_list)):
                BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
                    self.var_list[num], self.teacher_running_var[num])

            BNS_loss = BNS_loss / len(self.mean_list)
            if epoch < 50:
                sim, sia_loss = self.train_simese(
                    images.clone().detach(), labels.detach(), output_teacher_batch.clone().detach())
                siamese_loss.append(sia_loss.cpu().item())

            idx = torch.nonzero(torch.argmax(
                output_teacher_batch, dim=1) == labels)
            diversity_loss = torch.tensor(0.).cuda()
            if idx.numel() > 0:
                sim = self.sim_predict(
                    images[idx].squeeze(1), labels[idx].squeeze(1))
                diversity_loss += (sim *
                                   (- torch.log(1 - sim+1e-4))).mean()

            if idx.numel() > 0:
                if epoch > 20:
                    loss_G = loss_one_hot + 0.1 * BNS_loss + 0.5 * diversity_loss
                else:
                    loss_G = loss_one_hot + 0.1 * BNS_loss
            else:
                loss_G = loss_one_hot + 0.1 * BNS_loss
            self.backward_G(loss_G)

            # if sim is not None:
            #     if epoch < 5:
            #         self.collect_passing_threshold(
            #             sim, images, labels, output_teacher_batch, threshold=1.2)
            #     else:
            #         self.collect_passing_threshold(
            #             sim, images, labels, output_teacher_batch, threshold=0.5)

            # images = self.img_batch[:self.settings.batchSize]
            # labels = self.label_batch[:self.settings.batchSize]
            # output_teacher_batch = self.output_teacher_batch[:self.settings.batchSize]

            output, loss_S = self.forward(
                images.detach(), output_teacher_batch.detach(), labels)

            if epoch >= self.settings.warmup_epochs:
                self.backward_S(loss_S)

            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output, labels=labels,
                loss=loss_S, top5_flag=True, mean_flag=True)

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))

            end_time = time.time()

            gt = labels.data.cpu().numpy()
            d_acc = np.mean(
                np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt)

            fp_acc.update(d_acc)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [One-hot loss: %f] [BNS_loss:%f] [S loss: %f]"
            % (epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, loss_G.item(), loss_one_hot.item(), BNS_loss.item(),
               loss_S.item())
        )
        print(
            f"Siamese loss{np.mean(siamese_loss)} diversity loss {diversity_loss.item()}")
        self.output_teacher_batch = None
        self.img_batch = None
        self.label_batch = None

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def test(self, epoch):
        """
        testing
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model.eval()
        self.model_teacher.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()

                labels = labels.cuda()
                images = images.cuda()
                output = self.model(images)

                loss = torch.ones(1)
                self.mean_list.clear()
                self.var_list.clear()

                single_error, single_loss, single5_error = utils.compute_singlecrop(
                    outputs=output, loss=loss,
                    labels=labels, top5_flag=True, mean_flag=True)

                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
            % (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
        )

        self.scalar_info['testing_top1error'] = top1_error.avg
        self.scalar_info['testing_top5error'] = top5_error.avg
        self.scalar_info['testing_loss'] = top1_loss.avg
        if self.tensorboard_logger is not None:
            for tag, value in self.scalar_info.items():
                self.tensorboard_logger.scalar_summary(
                    tag, value, self.run_count)
            self.scalar_info = {}
        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def test_teacher(self, epoch):
        """
        testing
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model_teacher.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()
                data_time = start_time - end_time

                labels = labels.cuda()
                if self.settings.tenCrop:
                    image_size = images.size()
                    images = images.view(
                        image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
                    images_tuple = images.split(image_size[0])
                    output = None
                    for img in images_tuple:
                        if self.settings.nGPU == 1:
                            img = img.cuda()
                        img_var = Variable(img, volatile=True)
                        temp_output, _ = self.forward(img_var)
                        if output is None:
                            output = temp_output.data
                        else:
                            output = torch.cat((output, temp_output.data))
                    single_error, single_loss, single5_error = utils.compute_tencrop(
                        outputs=output, labels=labels)
                else:
                    if self.settings.nGPU == 1:
                        images = images.cuda()

                    output = self.model_teacher(images)

                    loss = torch.ones(1)
                    self.mean_list.clear()
                    self.var_list.clear()

                    single_error, single_loss, single5_error = utils.compute_singlecrop(
                        outputs=output, loss=loss,
                        labels=labels, top5_flag=True, mean_flag=True)
                #
                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

        print(
            "Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
            % (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
        )

        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg
