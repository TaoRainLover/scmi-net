import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, tao=1e-2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        # 计算欧氏距离的平方
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True).pow(2)

        # 计算对比损失
        contrastive_loss = torch.mean(
            (1 - target) * euclidean_distance + target * torch.clamp(self.margin - euclidean_distance, min=0))

        return contrastive_loss


class DecoupleLoss(nn.Module):
    def __init__(self, tau=10):
        super(DecoupleLoss, self).__init__()
        self.tau = tau

    def forward(self, pe_s, pe_t, pc_s, pc_t):
        numerator_s_t = torch.exp(pe_s.dot(pe_t) / self.tau)
        denominator_s_t = torch.exp(pe_s.dot(pc_s) / self.tau) + torch.exp(pe_s.dot(pc_t) / self.tau)

        numerator_t_s = torch.exp(pe_t.dot(pe_s) / self.tau)
        denominator_t_s = torch.exp(pe_t.dot(pc_t) / self.tau) + torch.exp(pe_t.dot(pc_s) / self.tau)

        loss = -0.5 * (torch.log(numerator_s_t / denominator_s_t) + torch.log(numerator_t_s / denominator_t_s))
        return loss

class DecoupleLoss_V2(nn.Module):
    def __init__(self, tau=0.5):
        super(DecoupleLoss_V2, self).__init__()
        self.tau = tau

    def forward(self, pe_s, pe_t, pc_s, pc_t):
        numerator_s_t = torch.exp(pe_s.dot(pe_t) / self.tau)
        # denominator_s_t = torch.exp(pe_s.dot(pc_s) / self.tau) + torch.exp(pe_s.dot(pc_t) / self.tau)
        denominator_s_t = torch.logsumexp(torch.stack([pe_s.dot(pc_s) / self.tau, pe_s.dot(pc_t) / self.tau]), dim=0)

        numerator_t_s = torch.exp(pe_t.dot(pe_s) / self.tau)
        # denominator_t_s = torch.exp(pe_t.dot(pc_t) / self.tau) + torch.exp(pe_t.dot(pc_s) / self.tau)
        denominator_t_s = torch.logsumexp(torch.stack([pe_t.dot(pc_t) / self.tau, pe_t.dot(pc_s) / self.tau]), dim=0)

        loss = -0.5 * (torch.log(numerator_s_t / denominator_s_t) + torch.log(numerator_t_s / denominator_t_s))
        return loss


# 多核最大均值误差（multiple kernels of maximum mean discrepancies, MK-MMD）
class MKMMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
            初始化
            Params:
             kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
             kernel_num: 取不同高斯核的数量
             fix_sigma: 是否固定，如果固定，则为单核MMD
        '''
        super(MKMMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        '''
            多核或单核高斯核矩阵函数，根据输入样本集x和y，计算返回对应的高斯核矩阵
            Params:
             source: (b1,n)的X分布样本数组
             target:（b2，n)的Y分布样本数组
             kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
             kernel_num: 取不同高斯核的数量
             fix_sigma: 是否固定，如果固定，则为单核MMD
            Return:
              sum(kernel_val): 多个核矩阵之和
        '''
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat((source, target), dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))

        L2_distance_square = torch.cumsum((total0 - total1).pow(2), dim=2)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance_square) / (n_samples**2 - n_samples)

        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]

        kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def forward(self, source, target):
        '''
            计算源域数据和目标域数据的MMD距离
            Params:
             source: (b1,n)的X分布样本数组
             target:（b2，n)的Y分布样本数组
            Return:
             loss: MK-MMD loss
        '''
        batch_size = int(source.size(0))
        kernels = self.gaussian_kernel(source, target)
        loss = 0
        # 将核矩阵分成4部分
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]

        # 这里计算出的n_loss是每个维度上的MK-MMD距离，一般还会做均值化处理
        n_loss = loss / float(batch_size)
        return torch.mean(n_loss)


class MKMMDLossV2(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
            初始化
            Params:
             kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
             kernel_num: 取不同高斯核的数量
             fix_sigma: 是否固定，如果固定，则为单核MMD
        '''
        super(MKMMDLossV2, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        '''
        将源域数据和目标域数据转化为核矩阵，即上文中的K
        Params:
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul:
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            sum(kernel_val): 多个核矩阵之和
        '''
        n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)#将source,target按列方向合并
        #将total复制（n+m）份
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total0-total1)**2).sum(2)
        #调整高斯核函数的sigma值
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        #高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        #得到最终的核矩阵
        return sum(kernel_val)#/len(kernel_val)

    def forward(self, source, target):
        '''
        计算源域数据和目标域数据的MMD距离
        Params:
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul:
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            loss: MMD loss
        '''
        batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
        kernels = self.gaussian_kernel(source, target)
        #根据式（3）将核矩阵分成4部分
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss#因为一般都是n==m，所以L矩阵一般不加入计算


if __name__ == '__main__':
    """test decouple loss """
    size = 128
    source_emotion_features = torch.rand(size)
    target_emotion_features = torch.rand(size)
    source_domain_features = torch.rand(size)
    target_domain_features = torch.rand(size)
    source_emotion_prototype = F.normalize(source_emotion_features, p=2, dim=0)
    target_emotion_prototype = F.normalize(target_emotion_features, p=2, dim=0)
    source_domain_prototype = F.normalize(source_domain_features, p=2, dim=0)
    target_domain_prototype = F.normalize(target_domain_features, p=2, dim=0)
    decouple_loss = DecoupleLoss()
    decouple_loss_v2 = DecoupleLoss_V2()
    loss = decouple_loss(source_emotion_prototype, target_emotion_prototype, source_domain_prototype, target_domain_prototype)
    print(f"loss: {loss}")
    print(f"loss: {decouple_loss_v2(source_emotion_prototype, target_emotion_prototype, source_domain_prototype, target_domain_prototype)}")

    """ 
    test mk-mmd loss 
    mk_mmd_loss = MKMMDLoss()
    mk_mmd_loss_v2 = MKMMDLossV2()

    source_data = torch.randn(100, 2)
    target_data = torch.randn(100, 2) + 0.01
    # target_data = source_data + 0

    # Calculate and print the MK-MMD loss
    print(f"MK-MMD Loss: {mk_mmd_loss(source_data, target_data)}")
    print(f"MK-MMD-V2 Loss: {mk_mmd_loss_v2(source_data, target_data)}")
    """
