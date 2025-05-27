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
