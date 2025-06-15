import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather

# ################# Normal LOSS####################
class HardNegativeNLLLoss:
    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        scores = self.similarity_fct(full_q_reps, d_reps) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )

        loss = self.cross_entropy_loss(scores, labels)
        return loss
# ################# Normal LOSS####################

################ DEBATER LOSS ####################
class DEBATERHardNegativeNLLLoss:
    def __init__(
            self,
            scale: float = 20.0,
            similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
            self,
            q_reps: Tensor,
            d_reps_pos: Tensor,
            d_reps_neg: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        #print("q_reps shape:", q_reps.shape) # [b,2304]
        #print("d_reps_pos shape:", d_reps_pos.shape)  # [b,8,2304]
        #print("d_reps_neg shape:", d_reps_neg.shape) #  # [b,8,2304]

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        ##### loss1 ######
        d_reps_pos_last8_loss1 = full_d_reps_pos[:,-1,:]
        d_reps_neg_last8_loss1 = full_d_reps_neg[:,-1,:]

        d_reps_last8_loss1 = torch.cat([d_reps_pos_last8_loss1, d_reps_neg_last8_loss1], dim=0)
        scores_loss1 = self.similarity_fct(full_q_reps, d_reps_last8_loss1) * self.scale
        labels_loss1 = torch.tensor(
                range(len(scores_loss1)), dtype=torch.long, device=scores_loss1.device
        )
        loss1 = self.cross_entropy_loss(scores_loss1, labels_loss1)
        ####### loss1 ######

        ####### loss2 ######
        d_reps_pos_last8 = full_d_reps_pos
        d_reps_neg_last8 = full_d_reps_neg

        d_reps_last8 = torch.cat([d_reps_pos_last8, d_reps_neg_last8], dim=0)
        q_b = full_q_reps.size(0)
        d_b = d_reps_last8.size(0)
        d_views = d_reps_last8.size(1)
        h_dim = d_reps_last8.size(2)

        d_reps_last8 = d_reps_last8.reshape(d_b * d_views, h_dim) # (b*2*8, 2304)
        sim_scores = self.similarity_fct(full_q_reps, d_reps_last8)  # (b, b*2*8)
        sim_scores = sim_scores.view(q_b,d_b,d_views)
        ensemble_scores = sim_scores.sum(dim = 2).reshape(q_b ,d_b) * self.scale
        teacher_targets = torch.softmax(ensemble_scores.detach(), dim=-1)
        loss2  = - torch.mean(
            torch.sum(torch.log_softmax(scores_loss1, dim=-1) * teacher_targets, dim=-1))
        ####### loss2 ######

        loss = (loss1 + loss2) / 2
        return loss
################## DEBATER LOSS ####################

# ################### MultiViews LOSS ####################
class MultiViewsHardNegativeNLLLoss:
    def __init__(
            self,
            scale: float = 20.0,
            similarity_fct=cos_sim,
            base: float = 1.2,  # ????
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.base = base  # ??????????????

    def __call__(self, q_reps: Tensor, d_reps_pos: Tensor, d_reps_neg: Tensor = None):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)
            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)
            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        d_reps_last8 = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        q_b = full_q_reps.size(0)
        d_b = d_reps_last8.size(0)
        d_views = d_reps_last8.size(1)
        h_dim = d_reps_last8.size(2)

        d_reps_last8 = d_reps_last8.reshape(d_b * d_views, h_dim)
        sim_scores = self.similarity_fct(full_q_reps, d_reps_last8)
        sim_scores = sim_scores.view(q_b, d_b, d_views)

        # ????????????????? tensor([0.0606, 0.0727, 0.0873, 0.1047, 0.1257, 0.1508, 0.1810, 0.2172])
        weights = torch.pow(self.base, torch.arange(d_views, device=sim_scores.device).float())
        # ????????????????1???????0.05??????????
        random_noise = torch.normal(mean=1.0, std=0.05, size=weights.size(), device=weights.device)
        weights = weights * random_noise  # ???????
        weights = weights / weights.sum()  # ???????????

        total_loss = 0.0
        for v in range(d_views):
            scores = sim_scores[:, :, v] * self.scale
            labels = torch.arange(len(scores), dtype=torch.long, device=scores.device)
            loss = self.cross_entropy_loss(scores, labels)
            total_loss += weights[v] * loss

        return total_loss

# ################# MRL-Enhanced HardNegativeNLLLoss ####################
class MRLHardNegativeNLLLoss:
    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
        sub_dims=None,
        weights=None,
    ):
        """
        基于原始 HardNegativeNLLLoss 的改造，支持 MRL 多子空间对比。
        
        参数:
          scale:             logits 缩放系数，原始实现里默认为 20.0。
          similarity_fct:    相似度函数，一般使用 cos_sim。
          sub_dims:          子空间维度列表（List[int]）。例如 [64, 256, 1024, 4096]。
                             如果为 None，则等价于只使用“完整向量”维度一次对比，
                             即 sub_dims=[D]，其中 D = q_reps.shape[-1]。
          weights:           每个子空间损失的加权权重（List[float]）。长度必须等于 sub_dims 长度。
                             如果为 None，则默认各子空间权重均等分配。
        """
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # 如果用户不传 sub_dims，就只用完整维度一次计算
        if sub_dims is None:
            # placeholder，会在第一次调用 __call__ 时确定实际维度
            self.sub_dims = None
        else:
            self.sub_dims = list(sub_dims)

        if weights is None:
            self.weights = None  # 在 __call__ 里按 sub_dims 数量平均赋值
        else:
            self.weights = list(weights)

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor = None,
    ):
        """
        q_reps:      [B, D]，查询向量
        d_reps_pos:  [B, D]，正例向量
        d_reps_neg:  [N, D]，负例向量。如果为 None，则视为空张量。

        返回:
          multi_loss: 多子空间加权后的总损失（标量）。
        """
        if d_reps_neg is None:
            # 构造一个形状正确但不含元素的空张量
            d_reps_neg = d_reps_pos[:0, :]

        # 分布式 all_gather 扩充正/负样本池，保持与原实现逻辑一致：
        if torch.distributed.is_initialized():
            gathered_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(gathered_pos, dim=0)

            gathered_q = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(gathered_q, dim=0)

            gathered_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(gathered_neg, dim=0)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        # 将正例、负例拼成候选集合 d_reps_pool，形状 [B_total + N_total, D]
        d_reps_pool = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)

        # 如果 sub_dims 为空，则在此时进行一次初始化：只用完整向量维度
        if self.sub_dims is None:
            D = full_q_reps.size(-1)
            self.sub_dims = [D]

        # 如果没有传入 weights，则平均分配
        if self.weights is None:
            self.weights = [1.0 / len(self.sub_dims)] * len(self.sub_dims)
        else:
            assert len(self.weights) == len(self.sub_dims), (
                f"weights 长度 ({len(self.weights)}) 必须等于 sub_dims 长度 ({len(self.sub_dims)})"
            )

        total_loss = 0.0
        # 对每个子空间进行对比损失计算
        for idx, dim in enumerate(self.sub_dims):
            # 逐子空间切片
            q_sub = full_q_reps[:, :dim]       # [B_total, dim]
            pool_sub = d_reps_pool[:, :dim]    # [B_total + N_total, dim]

            # 计算相似度矩阵: [B_total, B_total + N_total]
            scores = self.similarity_fct(q_sub, pool_sub) * self.scale

            # 构造 labels = [0,1,...,B_total-1]
            batch_size = q_sub.size(0)
            labels = torch.arange(batch_size, dtype=torch.long, device=scores.device)

            # 交叉熵
            loss_sub = self.cross_entropy_loss(scores, labels)
            total_loss += self.weights[idx] * loss_sub

        return total_loss

