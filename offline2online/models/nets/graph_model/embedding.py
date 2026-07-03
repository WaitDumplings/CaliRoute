import torch
import torch.nn as nn

def AutoEmbedding(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "evrptw": EVRPTWEmbedding,
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding

# Embedding Layer for EVRPTW
class EVRPTWEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = None,
        use_svd_distance_embedding: bool = False,
        rdi_svd_rank: int = 10,
        rdi_svd_feature_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embedding_dim
        self.use_svd_distance_embedding = bool(use_svd_distance_embedding)
        self.rdi_svd_rank = max(1, int(rdi_svd_rank))
        self.rdi_svd_feature_dim = int(rdi_svd_feature_dim or embedding_dim)

        # raw feature dim = 2(x,y) + 1(demand) + 2(tw) + 1(service_time) = 6
        in_dim = 6 + (self.rdi_svd_feature_dim if self.use_svd_distance_embedding else 0)
        if self.use_svd_distance_embedding:
            self.svd_proj = nn.Sequential(
                nn.LayerNorm(2 * self.rdi_svd_rank),
                nn.Linear(2 * self.rdi_svd_rank, self.rdi_svd_feature_dim),
                nn.SiLU(),
                nn.Linear(self.rdi_svd_feature_dim, self.rdi_svd_feature_dim),
            )
        else:
            self.svd_proj = None

        # Type-specific projections
        self.depot_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, embedding_dim),
        )

        self.customer_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, embedding_dim),
        )

        self.rs_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, embedding_dim),
        )

        # 0: depot, 1: RS, 2: customer
        self.type_embed = nn.Embedding(3, embedding_dim)
        self.post_fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.depot_proj, self.customer_proj, self.rs_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        nn.init.zeros_(self.type_embed.weight)

    def _ensure_depot_shape(self, depot_loc: torch.Tensor) -> torch.Tensor:
        # Accept either [B,2] or [B,1,2]
        if depot_loc.dim() == 2:
            depot_loc = depot_loc.unsqueeze(1)
        return depot_loc

    def _svd_node_features(self, x: dict, node_count: int) -> torch.Tensor:
        dist = x.get("rdi_distance_matrix")
        if dist is None:
            dist = x.get("edge_distance")
        if dist is None:
            raise KeyError("SVD distance embedding requires rdi_distance_matrix or edge_distance in observations")
        if dist.dim() == 2:
            dist = dist.unsqueeze(0)
        dist = dist.to(dtype=torch.float32)
        if dist.size(-1) != node_count or dist.size(-2) != node_count:
            raise ValueError(
                f"SVD distance matrix shape {tuple(dist.shape)} does not match node count {node_count}"
            )
        finite = torch.isfinite(dist)
        safe = torch.where(finite, dist, torch.zeros_like(dist))
        denom = finite.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
        mean = safe.sum(dim=(-2, -1), keepdim=True) / denom
        centered = torch.where(finite, dist - mean, torch.zeros_like(dist))
        var = (centered * centered).sum(dim=(-2, -1), keepdim=True) / denom
        normalized = centered / var.sqrt().clamp_min(1e-6)
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        rank = min(self.rdi_svd_rank, int(normalized.size(-1)), int(normalized.size(-2)))
        try:
            u, s, v = torch.svd_lowrank(normalized, q=rank, niter=2)
        except (RuntimeError, NotImplementedError):
            u, s, vh = torch.linalg.svd(normalized, full_matrices=False)
            u = u[..., :rank]
            s = s[..., :rank]
            v = vh.transpose(-2, -1)[..., :rank]
        root_s = s.clamp_min(0.0).sqrt()
        source = u[..., :rank] * root_s.unsqueeze(-2)
        target = v[..., :rank] * root_s.unsqueeze(-2)
        svd = torch.cat([source, target], dim=-1)
        target_width = 2 * self.rdi_svd_rank
        if svd.size(-1) < target_width:
            svd = torch.nn.functional.pad(svd, (0, target_width - svd.size(-1)))
        return self.svd_proj(svd.to(next(self.parameters()).dtype))

    def forward(self, x):
        """
        Build node embeddings in fixed order: [depot, customers, RS].
        """
        depot_loc = self._ensure_depot_shape(x["depot_loc"])   # [B,1,2]
        cus_loc = x["cus_loc"]                                 # [B,n_cus,2]
        rs_loc = x["rs_loc"]                                   # [B,n_rs,2]

        B = depot_loc.size(0)
        device = depot_loc.device
        n_cus = cus_loc.size(1)
        n_rs = rs_loc.size(1)

        demand = x["demand"]                                   # [B,N,1]
        time_window = x["time_window"]                         # [B,N,2]
        service_time = x["service_time"]                       # [B,N,1]
        svd_feat = None
        if self.use_svd_distance_embedding:
            svd_feat = self._svd_node_features(x, 1 + n_cus + n_rs).to(
                device=device,
                dtype=depot_loc.dtype,
            )

        # ----- depot -----
        depot_demand = demand[:, :1, :]
        depot_tw = time_window[:, :1, :]
        depot_service = service_time[:, :1, :]
        depot_feat = torch.cat(
            [depot_loc, depot_demand, depot_tw, depot_service], dim=-1
        )  # [B,1,6]
        if svd_feat is not None:
            depot_feat = torch.cat([depot_feat, svd_feat[:, :1, :]], dim=-1)

        # ----- customers -----
        cus_demand = demand[:, 1:1 + n_cus, :]
        cus_tw = time_window[:, 1:1 + n_cus, :]
        cus_service = service_time[:, 1:1 + n_cus, :]
        cus_feat = torch.cat(
            [cus_loc, cus_demand, cus_tw, cus_service], dim=-1
        )  # [B,n_cus,6]
        if svd_feat is not None:
            cus_feat = torch.cat([cus_feat, svd_feat[:, 1:1 + n_cus, :]], dim=-1)

        # ----- RS -----
        rs_demand = demand[:, 1 + n_cus:, :]
        rs_tw = time_window[:, 1 + n_cus:, :]
        rs_service = service_time[:, 1 + n_cus:, :]
        rs_feat = torch.cat(
            [rs_loc, rs_demand, rs_tw, rs_service], dim=-1
        )  # [B,n_rs,6]
        if svd_feat is not None:
            rs_feat = torch.cat([rs_feat, svd_feat[:, 1 + n_cus:, :]], dim=-1)

        # Type-specific projections
        depot_emb = self.depot_proj(depot_feat)       # [B,1,D]
        cus_emb = self.customer_proj(cus_feat)        # [B,n_cus,D]
        rs_emb = self.rs_proj(rs_feat)                # [B,n_rs,D]

        node_emb = torch.cat([depot_emb, cus_emb, rs_emb], dim=1)  # [B,N,D]

        # type embedding
        depot_type = torch.zeros(B, 1, dtype=torch.long, device=device)          # 0
        cus_type = torch.full((B, n_cus), 2, dtype=torch.long, device=device)    # 2
        rs_type = torch.ones(B, n_rs, dtype=torch.long, device=device)            # 1
        node_type = torch.cat([depot_type, cus_type, rs_type], dim=1)            # [B,N]

        type_emb = self.type_embed(node_type)                                     # [B,N,D]

        node_emb = node_emb + type_emb
        node_emb = node_emb + self.post_fusion(node_emb)
        node_emb = self.norm(node_emb)

        return node_emb
