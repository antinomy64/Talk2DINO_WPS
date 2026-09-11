import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.model import CLIPLastLayer

class Contrastive(nn.Module):
    def __init__(self, sim=None, margin=0, max_violation=False, ltype='triplet'):
        super(Contrastive, self).__init__()
        self.margin = margin
        self.sim = sim
        self.max_violation = max_violation
        self.ltype = ltype
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_contrastive_loss(self, scores):
        if self.ltype == 'infonce':
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * scores
            logits_per_text = logits_per_image.t()

            # compute bidirectional CE loss
            num_logits = logits_per_image.shape[0]
            labels = torch.arange(num_logits, device=logits_per_image.device, dtype=torch.long)
            loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
                ) / 2

        elif self.ltype == 'triplet':
            diagonal = scores.diag().view(scores.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)

            # compare every diagonal score to scores in its column
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            # compare every diagonal score to scores in its row
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)

            # clear diagonals
            mask = torch.eye(scores.size(0)) > .5
            I = mask
            if torch.cuda.is_available():
                I = I.to(scores.device)
            cost_s = cost_s.masked_fill_(I, 0)
            cost_im = cost_im.masked_fill_(I, 0)

            # keep the maximum violating negative for each query
            if self.max_violation:
                cost_s = cost_s.max(1)[0]
                cost_im = cost_im.max(0)[0]

            loss = cost_s.sum() + cost_im.sum()
            
        else:
            raise ValueError(f'{self.ltype} not known!')
            
        return loss / scores.shape[0]**2 # normalization by the batch size**2

class ContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, sim, margin=0, max_violation=False, ltype='triplet'):
        super(ContrastiveLoss, self).__init__(sim=sim, margin=margin, max_violation=max_violation, ltype=ltype)
        

    def forward(self, im, s, return_similarity_mat=False, self_attn_maps=None, cls=None, text_input_mask=None, text_argmax=None, return_index=False):
        # compute image-sentence score matrix
        if type(self.sim) == CLIPLastLayer:
            scores = self.sim(im, s, ret_similarity_matrix=True, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, text_argmax=text_argmax)
        else:
            if return_index:
                scores, index = self.sim(im, s, ret_similarity_matrix=True, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, return_index=return_index)
            else:
                scores = self.sim(im, s, ret_similarity_matrix=True, self_attn_maps=self_attn_maps, cls=cls, text_input_mask=text_input_mask, return_index=return_index)
        loss = self.compute_contrastive_loss(scores)
        
        to_return = [loss]
        if return_similarity_mat:
            to_return.append(scores)
        if return_index:
            to_return.append(index)
        if len(to_return) > 1:
            to_return = tuple(to_return)
        else:
            to_return = to_return[0]
        return to_return

# ============================================================
# Part semantic structure preservation
# ============================================================

import torch
from torch import nn
from torch.nn import functional as F

class PartStructureRankLoss(nn.Module):
    """
    Object-wise part-relation rank preservation loss.

    Soft-rank formula is retained; projector input is now raw CLIP.
    Output normalization matches project_raw_clip_bank (eps=1e-12).

    Optimization only:
      1) project the full fixed part bank once per forward;
      2) cache raw-side pairwise soft-rank targets once in __init__;
      3) cache object row indices and upper-triangle indices as buffers.

    For every valid object o:
        L_o = 1 - corr(
            SoftRank(upper(cos(projected_parts_o))),
            SoftRank(upper(cos(raw_parts_o)))
        )

    Final loss is the equal-weight mean over valid objects.
    """

    def __init__(
        self,
        raw_part_features,
        object_groups,
        rank_temperature=0.05,
        min_parts=3,
        eps=1e-6,
    ):
        super().__init__()

        if rank_temperature <= 0:
            raise ValueError("rank_temperature must be > 0")

        if min_parts < 3:
            raise ValueError("min_parts must be >= 3")

        raw_part_features = torch.as_tensor(
            raw_part_features,
            dtype=torch.float32,
        )

        if raw_part_features.ndim != 2:
            raise ValueError(
                "raw_part_features must be [N_parts, D], "
                f"got {tuple(raw_part_features.shape)}"
            )

        # Raw prompt means are the actual input used by the W trainer.
        raw_part_features = raw_part_features.detach().clone()
        if not torch.isfinite(raw_part_features).all():
            raise ValueError("Nonfinite raw CLIP bank")
        if (raw_part_features.norm(dim=-1) == 0).any():
            raise ValueError("Zero raw CLIP feature")
        raw_for_similarity = F.normalize(raw_part_features, dim=-1, eps=eps)

        self.register_buffer(
            "raw_part_features",
            raw_part_features.contiguous(),
        )

        if isinstance(object_groups, dict):
            groups = [
                (str(name), ids)
                for name, ids in object_groups.items()
            ]
        else:
            groups = [
                (str(i), ids)
                for i, ids in enumerate(object_groups)
            ]

        self.object_names = []
        valid_groups = []

        for name, ids in groups:
            ids = torch.as_tensor(
                ids,
                dtype=torch.long,
            ).reshape(-1)

            if ids.numel() < min_parts:
                continue

            if int(ids.min()) < 0 or int(ids.max()) >= raw_part_features.shape[0]:
                raise IndexError(
                    f"group {name!r} contains row outside "
                    f"[0, {raw_part_features.shape[0]})"
                )

            self.object_names.append(name)
            valid_groups.append((name, ids))

        if len(valid_groups) == 0:
            raise ValueError(
                "No object group contains enough parts."
            )

        self.rank_temperature = float(rank_temperature)
        self.min_parts = int(min_parts)
        self.eps = float(eps)
        self.num_object_groups = len(valid_groups)

        # Cache all fixed/raw-side quantities once.
        for group_index, (_, ids) in enumerate(valid_groups):
            self.register_buffer(
                f"group_ids_{group_index}",
                ids.contiguous(),
            )

            k = int(ids.numel())
            tri = torch.triu_indices(
                k,
                k,
                offset=1,
            )

            self.register_buffer(
                f"tri_row_{group_index}",
                tri[0].contiguous(),
            )
            self.register_buffer(
                f"tri_col_{group_index}",
                tri[1].contiguous(),
            )

            raw = raw_for_similarity.index_select(0, ids)
            raw_similarity = raw @ raw.t()

            raw_values = raw_similarity[
                tri[0],
                tri[1],
            ]

            raw_rank = self._soft_rank(
                raw_values,
                self.rank_temperature,
            ).detach()

            self.register_buffer(
                f"raw_rank_{group_index}",
                raw_rank.contiguous(),
            )

    @staticmethod
    def _soft_rank(
        values,
        temperature,
    ):
        """
        Ascending differentiable soft rank.

        Larger similarity -> larger rank.
        """

        if values.ndim != 1:
            raise ValueError(
                "soft_rank input must be 1-D"
            )

        differences = (
            values[:, None]
            - values[None, :]
        ) / float(temperature)

        return (
            1.0
            + torch.sigmoid(
                differences
            ).sum(dim=1)
            - 0.5
        )

    @staticmethod
    def _correlation_loss(
        source,
        target,
        eps=1e-6,
    ):
        """
        1 - Pearson correlation of soft-rank vectors.
        """

        source = source - source.mean()
        target = target - target.mean()

        source = source / (
            source.norm().clamp_min(eps)
        )

        target = target / (
            target.norm().clamp_min(eps)
        )

        correlation = (
            source * target
        ).sum()

        return 1.0 - correlation

    def forward(
        self,
        projector,
        return_per_object=False,
    ):
        """
        Project raw CLIP once, matching the initial text used by RelProto.
        """

        # Previous implementation projected each object's rows separately.
        # ProjectionLayer is row-wise, so batching all rows is mathematically
        # identical while avoiding many tiny CUDA launches.
        projected_all = projector.project_clip_txt(
            self.raw_part_features
        )

        projected_all = F.normalize(
            projected_all.float(),
            dim=-1,
            eps=1e-12,
        )

        losses = []
        per_object = {}

        for group_index, object_name in enumerate(
            self.object_names
        ):
            ids = getattr(
                self,
                f"group_ids_{group_index}",
            )
            tri_row = getattr(
                self,
                f"tri_row_{group_index}",
            )
            tri_col = getattr(
                self,
                f"tri_col_{group_index}",
            )
            raw_rank = getattr(
                self,
                f"raw_rank_{group_index}",
            )

            projected = projected_all.index_select(
                0,
                ids,
            )

            projected_similarity = (
                projected @ projected.t()
            )

            projected_values = projected_similarity[
                tri_row,
                tri_col,
            ]

            projected_rank = self._soft_rank(
                projected_values,
                self.rank_temperature,
            )

            loss_o = self._correlation_loss(
                projected_rank,
                raw_rank,
                eps=self.eps,
            )

            losses.append(loss_o)

            if return_per_object:
                per_object[
                    object_name
                ] = (
                    1.0
                    - loss_o.detach()
                )

        loss = torch.stack(
            losses
        ).mean()

        if return_per_object:
            return loss, per_object

        return loss

    @classmethod
    def from_file(
        cls,
        path,
        rank_temperature=0.05,
        min_parts=3,
        eps=1e-6,
    ):
        """
        Expected file format:

        {
            "features": Tensor[N_parts, D],
            "object_groups": {
                "aeroplane": [...],
                "bird": [...],
                ...
            }
        }
        """

        data = torch.load(
            path,
            map_location="cpu",
        )

        if data.get("normalized", False) is not False:
            raise ValueError("Expected raw CLIP bank: normalized=False")

        if "features" in data:
            features = data["features"]

        elif "raw_features" in data:
            features = data["raw_features"]

        else:
            raise KeyError(
                "Structure bank must contain "
                "'features' or 'raw_features'."
            )

        if "object_groups" not in data:
            raise KeyError(
                "Structure bank must contain "
                "'object_groups'."
            )

        return cls(
            raw_part_features=features,
            object_groups=data[
                "object_groups"
            ],
            rank_temperature=(
                rank_temperature
            ),
            min_parts=min_parts,
            eps=eps,
        )

    @torch.no_grad()
    def exact_audit(self, projector):
        import numpy as np
        from scipy.stats import spearmanr
        projected = F.normalize(
            projector.project_clip_txt(self.raw_part_features).float(),
            dim=-1, eps=1e-12,
        )
        raw_unit = F.normalize(self.raw_part_features, dim=-1, eps=self.eps)
        values = {}
        for i, name in enumerate(self.object_names):
            ids = getattr(self, f"group_ids_{i}")
            r = getattr(self, f"tri_row_{i}")
            c = getattr(self, f"tri_col_{i}")
            x, y = raw_unit[ids], projected[ids]
            before = (x @ x.T)[r, c].cpu().numpy()
            after = (y @ y.T)[r, c].cpu().numpy()
            rho = float(spearmanr(before, after)[0])
            if not np.isfinite(rho):
                raise ValueError(f"Undefined exact Spearman for {name}")
            values[name] = rho
        values["macro"] = float(np.mean(list(values.values())))
        return values



def main():
    pass
    
if __name__ == '__main__':
    main()