# models.py
import math
from typing import Tuple, Optional

import torch
from torch import nn

def sample_masked_categorical(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    logp: (B,I,V) log-probs AFTER masked_log_softmax
    mask: (B,I,V) boolean feasibility mask used to create logp
    Returns chosen indices j in {0..V-1} as (B,I).
    Ensures depot fallback if a row was all-masked pre-softmax.
    """
    # Identify any row that still has -inf everywhere (shouldn’t happen after ensure_row_has_depot)
    all_neg_inf = ~torch.isfinite(logp).any(dim=-1)  # (B,I)
    if all_neg_inf.any():
        # Force depot=0 logp = 0; others -inf
        lp = torch.full_like(logp, float("-inf"))
        lp[..., 0] = 0.0
        logp = torch.where(all_neg_inf.unsqueeze(-1), lp, logp)

    # sample
    probs = logp.exp()  # safe: masked_log_softmax made it valid
    j = torch.distributions.Categorical(probs=probs).sample()  # (B,I)
    return j

def ensure_row_has_depot(mask: torch.Tensor, depot_idx: int = 0) -> torch.Tensor:
    """
    If a row (B,K,*) or (B,D,*) has no True, force depot_idx True on that row.
    Works for mask shape (B, I, V) where I=K or D.
    """
    assert mask.dim() == 3, f"mask must be (B,I,V), got {mask.shape}"
    B, I, V = mask.shape
    bad = ~mask.any(dim=-1)          # (B, I)
    if bad.any():
        mask = mask.clone()
        # zero the whole row, then set depot to True
        mask[bad] = False
        b_idx, i_idx = bad.nonzero(as_tuple=True)
        mask[b_idx, i_idx, depot_idx] = True
    return mask

def assert_same_shape(A: torch.Tensor, B: torch.Tensor, msg: str):
    assert A.shape == B.shape, f"{msg}: {A.shape} vs {B.shape}"

# ---------- small helpers ----------
def _normalize_weights(w: torch.Tensor) -> torch.Tensor:
    """
    Clamp (w1,w2) to >=0 and renormalize to sum=1.
    w: (B,2)
    """
    w = torch.clamp(w, min=0)
    denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return w / denom


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log-softmax over feasible entries only.
    mask: 1 for feasible, 0 for infeasible, same shape as logits along 'dim'.
    Falls back to uniform over feasible entries if logits are -inf after masking.
    """
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(mask == 0, neg_inf)
    # numerical guard: if a row is all -inf (no feasible), make the depot (idx 0) feasible as last resort
    if masked_logits.dim() >= 2:
        # reduce along target dim to detect -inf rows
        max_keep = torch.amax(masked_logits, dim=dim, keepdim=True)
        bad_rows = torch.isinf(max_keep) & (max_keep < 0)  # all -inf
        if bad_rows.any():
            # construct an index slice to set depot feasible with zero logit
            # We'll assume 'dim' is the last (common in this code); adapt otherwise.
            if dim != masked_logits.dim() - 1:
                raise ValueError("This guard assumes dim is the last dimension.")
            depot_idx = 0
            masked_logits = masked_logits.clone()
            masked_logits[bad_rows.expand_as(masked_logits)] = neg_inf
            masked_logits[..., depot_idx] = torch.where(
                bad_rows.squeeze(-1),
                torch.zeros_like(masked_logits[..., depot_idx]),
                masked_logits[..., depot_idx],
            )
    return torch.log_softmax(masked_logits, dim=dim)




class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden: int = 0, act=nn.GELU):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                act(),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)


# ---------- attention blocks ----------
class CrossAttention(nn.Module):
    """Multihead attention wrapper with shape convention (B, L, d)."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: Optional[torch.Tensor] = None):
        if v is None:
            v = k
        q,k,v = q.clamp(min=-20.0, max=20.0), k.clamp(min=-20.0, max=20.0), v.clamp(min=-20.0, max=20.0)
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = self.norm1(q + attn_out)
        y = self.norm2(x + self.ffn(x))
        return y


class SelfEncoder(nn.Module):
    """Transformer-style self-encoder over tokens (e.g., nodes)."""
    def __init__(self, d_model: int, nhead: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-5,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):  # (B, L, d)
        return self.enc(x)


class WADRLPolicy(nn.Module):
    """
    Two-head WADRL policy (truck & drone) with optional vehicle gate and weight-token conditioning.

    Inputs:
      graph_ctx : (B, N+1, Fg)     # depot+customers nodes (encoded features)
      trucks_ctx: (B, K,   Ft)     # per-truck features/tokens
      drones_ctx: (B, D,   Fd)     # per-drone  features/tokens
      mask_trk  : (B, K, N+1)      # 1=feasible node for truck k
      mask_dr   : (B, D, N+1)      # 1=feasible node for drone d
      w_pair    : (B, 2)           # (w1, w2) for F1/F2 scalarization

    Outputs:
      trk_logits: (B, K, N+1)      # raw truck-node logits
      dr_logits : (B, D, N+1)      # raw drone-node logits
      gate_logits (optional): (B,2)
      value: (B,1)

      Plus convenience masked log-probs (node-level) for sampling/loss if you want:
      trk_logp: (B, K, N+1)
      dr_logp : (B, D, N+1)
    """
    def __init__(self, Fg, Ft, Fd, d_model=256, nhead=8, depth=4, use_gate=True, depth_t = 8, depth_d = 8, depth_c = 8):
        super().__init__()
        self.use_gate = use_gate

        # projectors
        self.graph_proj  = MLP(Fg, d_model)
        self.truck_proj  = MLP(Ft, d_model)
        self.drone_proj  = MLP(Fd, d_model)
        self.w_proj      = MLP(2, d_model)  # weight token

        # encoders
        self.graph_enc   = SelfEncoder(d_model, nhead, depth)
        self.truck_enc   = SelfEncoder(d_model, nhead, depth_t)
        self.drone_enc   = SelfEncoder(d_model, nhead, depth_d)
        self.ctx_mix     = SelfEncoder(d_model, nhead, depth=depth_c)

        # cross-attention (queries are per-vehicle instances; keys/values are ctx pool)
        self.trk_cross   = CrossAttention(d_model, nhead)
        self.dr_cross    = CrossAttention(d_model, nhead)

        # node readout for dot-product scoring
        self.node_key    = nn.Linear(d_model, d_model)

        # vehicle gate (include weight token)
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(3 * d_model, d_model), nn.GELU(),
                nn.Linear(d_model, 2)
            )

        # critic: global pooled node enc + weight token
        self.critic = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        graph_ctx: torch.Tensor,
        trucks_ctx: torch.Tensor,
        drones_ctx: torch.Tensor,
        mask_trk: torch.Tensor,
        mask_dr: torch.Tensor,
        w_pair: torch.Tensor,
    ):
        B, V, _ = graph_ctx.shape
        _, K, _ = trucks_ctx.shape
        _, D, _ = drones_ctx.shape

        # --- 1) normalize weights & project tokens ---
        w_pair = _normalize_weights(w_pair)                 # (B,2)
        H = self.graph_proj(graph_ctx)                      # (B,V,d)
        T = self.truck_proj(trucks_ctx)                     # (B,K,d)
        T = self.truck_enc(T)
        R = self.drone_proj(drones_ctx)                     # (B,D,d)
        R = self.drone_enc(R)
        W_tok = self.w_proj(w_pair).unsqueeze(1)            # (B,1,d)

        # --- 2) encode nodes & mix context pool (nodes + pooled vehicle summaries + weight token) ---
        H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)
        H = H.clamp(min=-20.0, max=20.0)
        H = self.graph_enc(H)                               # (B,V,d)
        T_pool = T.mean(dim=1, keepdim=True)                # (B,1,d)
        R_pool = R.mean(dim=1, keepdim=True)                # (B,1,d)
        # print(H.shape, T_pool.shape, R_pool.shape, W_tok.shape)
        ctx_pool = torch.cat([H, T_pool, R_pool, W_tok], dim=1)  # (B,V+3,d)
        ctx_pool = torch.nan_to_num(ctx_pool, nan=0.0, posinf=1e6, neginf=-1e6)
        ctx_pool = ctx_pool.clamp(min=-20.0, max=20.0)
        ctx_pool = self.ctx_mix(ctx_pool)                   # (B,V+3,d)

        # --- 3) cross-attend per-vehicle instance queries over pool ---
        Tq = self.trk_cross(T, ctx_pool, ctx_pool)          # (B,K,d)
        Rq = self.dr_cross(R, ctx_pool, ctx_pool)           # (B,D,d)

        # --- 4) node-level logits via query·key ---
        node_keys = self.node_key(H)                        # (B,V,d)

        trk_logits = torch.einsum('bkd,bvd->bkv', Tq, node_keys)   # (B,K,V)
        dr_logits  = torch.einsum('bqd,bvd->bqv', Rq, node_keys)   # (B,D,V)

        trk_logits = torch.nan_to_num(trk_logits, nan=0.0, posinf=1e9, neginf=-1e9)
        dr_logits  = torch.nan_to_num(dr_logits , nan=0.0, posinf=1e9, neginf=-1e9)

        # --- 5) masked node log-probs (convenience) ---
        # mask_* must be broadcastable and boolean or {0,1} float.
        if mask_trk.dtype != torch.bool:
            mask_trk_bool = mask_trk > 0.5
        else:
            mask_trk_bool = mask_trk
        if mask_dr.dtype != torch.bool:
            mask_dr_bool = mask_dr > 0.5
        else:
            mask_dr_bool = mask_dr

        assert_same_shape(trk_logits, mask_trk_bool, "truck logits/mask mismatch")
        assert_same_shape(dr_logits, mask_dr_bool , "drone logits/mask mismatch")

        # 2) Make sure every row has at least one feasible action (fallback to depot=0)
        mask_trk_bool = ensure_row_has_depot(mask_trk_bool, depot_idx=0)
        mask_dr_bool  = ensure_row_has_depot(mask_dr_bool , depot_idx=0)

        trk_logp = masked_log_softmax(trk_logits, mask_trk_bool, dim=-1)  # (B,K,V)
        dr_logp = masked_log_softmax(dr_logits, mask_dr_bool , dim=-1)  # (B,D,V)


        # --- 6) optional vehicle gate (condition on W as well) ---
        gate_logits = None
        if self.use_gate:
            t_sum = Tq.mean(dim=1)                              # (B,d)
            r_sum = Rq.mean(dim=1)                              # (B,d)
            gate_in = torch.cat([t_sum, r_sum, W_tok.squeeze(1)], dim=-1)
            gate_in = torch.nan_to_num(gate_in, nan=0.0, posinf=1e9, neginf=-1e9)  # NEW
            gate_logits = self.gate(gate_in)

        # --- 7) critic ---
        global_feat = H.mean(dim=1)                             # (B,d)
        value = self.critic(torch.cat([global_feat, W_tok.squeeze(1)], dim=-1))  # (B,1)

        return {
            "trk_logits": trk_logits,
            "dr_logits":  dr_logits,
            "trk_logp":   trk_logp,
            "dr_logp":    dr_logp,
            "gate_logits": gate_logits,
            "value":      value,
        }

    @torch.no_grad()
    def sample_action(
        self,
        out: dict,
        mask_trk: torch.Tensor,
        mask_dr: torch.Tensor,
        use_gate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a composite action (veh_type, veh_idx, node_j) from model outputs and masks.
        Returns tensors of shape (B,) each:
          veh_type in {0=truck, 1=drone}
          veh_idx  in [0..K-1] or [0..D-1] depending on type
          node_j   in [0..N] (0 = depot)
        """
        trk_logp = out["trk_logp"]  # (B,K,V)
        dr_logp  = out["dr_logp"]   # (B,D,V)
        gate_logits = out.get("gate_logits", None)

        B, K, V = trk_logp.shape
        D = dr_logp.shape[1]

        # pick type
        if use_gate and gate_logits is not None:
            p_type = torch.softmax(gate_logits, dim=-1)             # (B,2)
        else:
            # If no gate, choose type proportional to aggregate feasible mass
            # (simple heuristic)
            p_trk = torch.softmax(trk_logp.logsumexp(dim=-1), dim=-1).mean(dim=-1, keepdim=True)  # (B,1)
            p_dr  = torch.softmax(dr_logp.logsumexp(dim=-1),  dim=-1).mean(dim=-1, keepdim=True)  # (B,1)
            p = torch.cat([p_trk, p_dr], dim=-1)
            p_type = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        veh_type = torch.distributions.Categorical(p_type).sample()  # 0=truck, 1=drone

        # pick instance and node within the chosen head
        veh_idx = torch.empty(B, dtype=torch.long, device=p_type.device)
        node_j  = torch.empty(B, dtype=torch.long, device=p_type.device)

        for b in range(B):
            if int(veh_type[b].item()) == 0:  # truck
                # choose instance by weighing each truck's feasible mass
                # instance probs ~ exp(logsumexp over feasible nodes)
                logp_nodes = trk_logp[b]  # (K,V)
                inst_logmass = logp_nodes.logsumexp(dim=-1)  # (K,)
                p_inst = torch.softmax(inst_logmass, dim=-1)
                k = torch.distributions.Categorical(p_inst).sample()
                j = torch.distributions.Categorical(torch.exp(logp_nodes[k])).sample()
                veh_idx[b] = k
                node_j[b]  = j
            else:  # drone
                logp_nodes = dr_logp[b]  # (D,V)
                inst_logmass = logp_nodes.logsumexp(dim=-1)  # (D,)
                p_inst = torch.softmax(inst_logmass, dim=-1)
                d = torch.distributions.Categorical(p_inst).sample()
                j = torch.distributions.Categorical(torch.exp(logp_nodes[d])).sample()
                veh_idx[b] = d
                node_j[b]  = j

        return veh_type, veh_idx, node_j
    
    # @torch.no_grad()
    # def act(self, obs: dict):
    #     # Expect obs keys as provided by env._build_obs()
    #     out = self.forward(
    #         obs["graph_ctx"], obs["trucks_ctx"], obs["drones_ctx"],
    #         obs["mask_trk"],  obs["mask_dr"],  obs["weights"]
    #     )
    #     B = out["value"].shape[0]
    #     # 1) sample vehicle type
    #     # veh_logp = out["veh_gate_logp"]              # [B,2] log-probs
    #     # veh_dist = torch.distributions.Categorical(logits=veh_logp)

    #     gate_logits = out["gate_logits"]  # [B,2] or None if use_gate=False
    #     if gate_logits is None:
    #         # fall back to 50/50
    #         veh_dist = torch.distributions.Categorical(logits=torch.zeros(out["value"].shape[0], 2, device=out["value"].device))
    #     else:
    #         veh_dist = torch.distributions.Categorical(logits=gate_logits)

    #     veh = veh_dist.sample()                      # [B]
    #     veh_logp_sel = veh_dist.log_prob(veh).unsqueeze(-1)  # [B,1]

    #     actions = []
    #     logps = []
    #     for b in range(B):
    #         if veh[b].item() == 0:
    #             # truck branch
    #             logp_trk = out["trk_logp"][b]   # [K,V]
    #             flat = logp_trk.view(-1)        # [K*V]
    #             dist = torch.distributions.Categorical(logits=flat)
    #             idx = dist.sample()
    #             logps.append(dist.log_prob(idx).unsqueeze(0))
    #             k = (idx // logp_trk.shape[1]).item()
    #             j = (idx %  logp_trk.shape[1]).item()
    #             actions.append((0, k, j))
    #         else:
    #             # drone branch
    #             logp_dr = out["dr_logp"][b]     # [D,V]
    #             flat = logp_dr.view(-1)         # [D*V]
    #             dist = torch.distributions.Categorical(logits=flat)
    #             idx = dist.sample()
    #             logps.append(dist.log_prob(idx).unsqueeze(0))
    #             d = (idx // logp_dr.shape[1]).item()
    #             j = (idx %  logp_dr.shape[1]).item()
    #             actions.append((1, d, j))

    #     act_logp = torch.stack(logps, dim=0) + veh_logp_sel  # [B,1] total logp
    #     value = out["value"]                                  # [B,1]
    #     return actions[0], act_logp, value, out  # B assumed 1 in rollout collector

    @torch.no_grad()
    def act(self, obs: dict):
        """
        Safe action sampler with gate override:
        - If gate chooses a branch with no feasible actions, fall back to the other branch.
        - If neither branch has feasible actions, return a no-op (0,0,0) or raise.
        Assumes B=1 in rollout collection.
        """
        out = self.forward(
            obs["graph_ctx"], obs["trucks_ctx"], obs["drones_ctx"],
            obs["mask_trk"],  obs["mask_dr"],  obs["weights"]
        )
        device = out["value"].device
        B = out["value"].shape[0]
        assert B == 1, "Collector assumes batch size 1."

        # --- Per-branch feasibility from masks ---
        # mask_* : (B, K/D, V), True = feasible
        mask_trk = obs["mask_trk"].to(device)  # (1,K,V)
        mask_dr  = obs["mask_dr"].to(device)   # (1,D,V)

        # Does each instance have at least one feasible node?
        inst_has_node_trk = mask_trk.any(dim=-1)     # (1,K)
        inst_has_node_dr  = mask_dr.any(dim=-1)      # (1,D)

        # Does the branch have any feasible instance?
        has_trk_moves = inst_has_node_trk.any().item()
        has_dr_moves  = inst_has_node_dr.any().item()

        # --- Sample vehicle type (with optional override) ---
        gate_logits = out["gate_logits"]  # (1,2) or None
        if gate_logits is None:
            veh_dist = torch.distributions.Categorical(logits=torch.zeros(1, 2, device=device))
        else:
            gate_logits = torch.nan_to_num(gate_logits, nan=0.0, posinf=0.0, neginf=0.0)
            veh_dist = torch.distributions.Categorical(logits=gate_logits)

        veh = veh_dist.sample()  # (1,)
        veh_logp_sel = veh_dist.log_prob(veh).unsqueeze(-1)  # (1,1)

        # Gate override when needed
        chosen = veh.item()  # 0 = truck, 1 = drone
        if chosen == 1 and not has_dr_moves and has_trk_moves:
            chosen = 0  # force truck
            # (optional) set veh_logp_sel to the log-prob of the forced choice:
            veh_logp_sel = veh_dist.log_prob(torch.tensor([0], device=device)).unsqueeze(-1)
        elif chosen == 0 and not has_trk_moves and has_dr_moves:
            chosen = 1  # force drone
            veh_logp_sel = veh_dist.log_prob(torch.tensor([1], device=device)).unsqueeze(-1)
        # elif not has_trk_moves and not has_dr_moves:
        #     # No moves anywhere: return a no-op or raise.
        #     # If your env treats (0,0,0) as a safe no-op at depot, do this:
        #     action = (0, 0, 0)
        #     # Total logp: use the gate’s probability mass for the chosen (arbitrary) type.
        #     # Here we just return a valid tensor; this situation should normally correspond to env "done".
        #     return action, torch.zeros(1,1, device=device), out["value"], out

        # --- Sample inside branch safely ---
        # NOTE: out["*_logp"] already come from masked_log_softmax (=-inf where infeasible)
        # We still need to guard an all-(-inf) row (shouldn't happen after gate override, but we’re safe).
        def sample_flat_from_logp(logp_2d, mask_2d):
            """
            logp_2d: (I,V), mask_2d: (I,V) booleans
            Returns (inst_idx, node_idx, logp_of_choice)
            """
            # Zero out fully infeasible rows
            row_has = mask_2d.any(dim=-1)  # (I,)
            if not row_has.any():
                return None

            # Flatten over only feasible rows to avoid sampling -inf only
            I, V = logp_2d.shape
            flat_logits = logp_2d.view(-1)  # (I*V,)
            flat_mask   = mask_2d.view(-1)  # (I*V,)

            if not flat_mask.any():
                return None

            # Replace -inf on infeasible entries; keep feasible logits as-is
            safe_logits = torch.where(flat_mask, flat_logits, torch.tensor(-float('inf'), device=flat_logits.device))
            # It’s still possible all feasible entries were numerically -inf; guard it:
            if torch.isneginf(safe_logits).all():
                return None

            dist = torch.distributions.Categorical(logits=safe_logits)
            idx  = dist.sample()
            logp = dist.log_prob(idx)
            inst = (idx // V).item()
            node = (idx %  V).item()
            return inst, node, logp

        if chosen == 0:
            # TRUCK
            trk_logp = out["trk_logp"][0]   # (K,V)
            ok = sample_flat_from_logp(trk_logp, mask_trk[0])
            if ok is None:
                # Fallback one more time to drones if possible
                if has_dr_moves:
                    chosen = 1
                else:
                    return (0,0,0), torch.zeros(1,1, device=device), out["value"], out
            else:
                k, j, lp = ok
                total_logp = lp.unsqueeze(0).unsqueeze(0) + veh_logp_sel  # (1,1)
                return (0, k, j), total_logp, out["value"], out

        # DRONE
        dr_logp = out["dr_logp"][0]        # (D,V)
        ok = sample_flat_from_logp(dr_logp, mask_dr[0])
        if ok is None:
            # Fallback to trucks if possible
            if has_trk_moves:
                trk_logp = out["trk_logp"][0]
                ok2 = sample_flat_from_logp(trk_logp, mask_trk[0])
                if ok2 is None:
                    return (0,0,0), torch.zeros(1,1, device=device), out["value"], out
                else:
                    k, j, lp = ok2
                    total_logp = lp.unsqueeze(0).unsqueeze(0) + veh_logp_sel
                    return (0, k, j), total_logp, out["value"], out
            else:
                return (0,0,0), torch.zeros(1,1, device=device), out["value"], out
        else:
            d, j, lp = ok
            total_logp = lp.unsqueeze(0).unsqueeze(0) + veh_logp_sel
            return (1, d, j), total_logp, out["value"], out


    @torch.no_grad()
    def value_only(self, obs: dict):
        out = self.forward(
            obs["graph_ctx"], obs["trucks_ctx"], obs["drones_ctx"],
            obs["mask_trk"],  obs["mask_dr"],  obs["weights"]
        )
        return out["value"]  # [B,1]

    def evaluate_actions(self, flat_obs: dict, actions: torch.Tensor):
        """
        Re-evaluate logp, entropy, and value for PPO.
        flat_obs has shapes [N, ...], actions is [N,3] of (veh, inst, node).
        Return:
          new_logp [N,1], entropy [N,1], value [N,1]
        """
        # --- inside WADRLPolicy.evaluate_actions(...):

        out = self.forward(
            flat_obs["graph_ctx"], flat_obs["trucks_ctx"], flat_obs["drones_ctx"],
            flat_obs["mask_trk"],  flat_obs["mask_dr"],  flat_obs["weights"]
        )

        N = actions.shape[0]
        new_logps = []
        ent_terms = []

        # ↓↓↓ FIX: use gate_logits instead of the old 'veh_gate_logp'
        gate_logits = out.get("gate_logits", None)
        # print("Gate logits : ",gate_logits)
        if gate_logits is None:
            gate_logits = torch.zeros(N, 2, device=out["value"].device)
        # else:
        gate_logits = torch.nan_to_num(gate_logits, nan=0.0)  # robustify
        veh_dist = torch.distributions.Categorical(logits=gate_logits)


        veh = actions[:, 0].long()
        veh_logp = veh_dist.log_prob(veh).unsqueeze(-1)  # [N,1]
        veh_ent  = veh_dist.entropy().unsqueeze(-1)      # [N,1]

        K = out["trk_logp"].shape[1]
        V = out["trk_logp"].shape[2]
        D = out["dr_logp"].shape[1]

        for n in range(N):
            if veh[n].item() == 0:
                # logp_trk = out["trk_logp"][n]     # [K,V] already masked log-probs
                # flat = logp_trk.view(-1)
                # idx = actions[n,1]*V + actions[n,2]
                # dist = torch.distributions.Categorical(logits=flat)

                # inside evaluate_actions loop (truck branch)
                logp_trk = out["trk_logp"][n]      # [K,V] log-probs (can contain -inf but not NaN)
                idx = actions[n,1]*V + actions[n,2]
                flat = logp_trk.reshape(-1)
                flat = torch.nan_to_num(flat, nan=-1e9)  # guard against any stray NaN
                # dist = torch.distributions.Categorical(logits=flat)
                # was: dist = torch.distributions.Categorical(logits=flat)
                probs = torch.exp(flat).clamp_min(1e-12)
                probs = probs / probs.sum()
                dist = torch.distributions.Categorical(probs=probs)


                new_logps.append(dist.log_prob(idx))
                ent_terms.append(dist.entropy())
            else:
                logp_dr = out["dr_logp"][n]       # [D,V] already masked log-probs
                flat = logp_dr.view(-1)
                flat = torch.nan_to_num(flat, nan=-1e9)
                idx = actions[n,1]*V + actions[n,2]
                # dist = torch.distributions.Categorical(logits=flat)
                # was: dist = torch.distributions.Categorical(logits=flat)
                probs = torch.exp(flat).clamp_min(1e-12)
                probs = probs / probs.sum()
                dist = torch.distributions.Categorical(probs=probs)

                new_logps.append(dist.log_prob(idx))
                ent_terms.append(dist.entropy())

        branch_logp = torch.stack(new_logps, dim=0).unsqueeze(-1)  # [N,1]
        branch_ent  = torch.stack(ent_terms, dim=0).unsqueeze(-1)  # [N,1]
        total_logp  = veh_logp + branch_logp
        entropy     = 0.5*veh_ent + 0.5*branch_ent
        value       = out["value"]
        return total_logp, entropy, value



# ---------------- sanity check ----------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # toy sizes
    B, N, K, D = 2, 5, 3, 4
    Fg, Ft, Fd = 7, 6, 9
    d_model = 128

    # fake inputs
    graph_ctx  = torch.randn(B, N+1, Fg)
    trucks_ctx = torch.randn(B, K, Ft)
    drones_ctx = torch.randn(B, D, Fd)

    # masks: ensure depot is always feasible (col 0)
    mask_trk = (torch.rand(B, K, N+1) > 0.2).to(torch.bool)
    mask_trk[..., 0] = True
    mask_dr  = (torch.rand(B, D, N+1) > 0.2).to(torch.bool)
    mask_dr[..., 0] = True

    # weight pairs (not normalized on purpose to test normalization)
    w_pair = torch.tensor([[0.2, 0.9], [1.5, 0.5]], dtype=torch.float32)
    print(w_pair.shape)

    policy = WADRLPolicy(Fg, Ft, Fd, d_model=d_model, nhead=4, depth=2, use_gate=True)

    out = policy(
        graph_ctx=graph_ctx,
        trucks_ctx=trucks_ctx,
        drones_ctx=drones_ctx,
        mask_trk=mask_trk,
        mask_dr=mask_dr,
        w_pair=w_pair
    )

    print("trk_logits:", out["trk_logits"].shape)  # (B,K,N+1)
    print("dr_logits :", out["dr_logits"].shape)   # (B,D,N+1)
    print("trk_logp  :", out["trk_logp"].shape)    # (B,K,N+1)
    print("dr_logp   :", out["dr_logp"].shape)     # (B,D,N+1)
    print("gate_logits:", None if out["gate_logits"] is None else out["gate_logits"].shape)
    print("value     :", out["value"].shape)

    # verify masked probabilities sum to 1 along nodes per instance
    trk_prob = torch.exp(out["trk_logp"])
    dr_prob  = torch.exp(out["dr_logp"])

    # sums over nodes
    s_trk = trk_prob.sum(dim=-1)  # (B,K)
    s_dr  = dr_prob.sum(dim=-1)   # (B,D)
    print("Truck node prob sums (should be 1 where mask has any True):")
    print(s_trk)
    print("Drone node prob sums (should be 1 where mask has any True):")
    print(s_dr)

    # sample a few actions
    veh_type, veh_idx, node_j = policy.sample_action(out, mask_trk, mask_dr, use_gate=True)
    print("Sampled type:", veh_type)   # 0=truck, 1=drone
    print("Sampled idx :", veh_idx)
    print("Sampled node:", node_j)
