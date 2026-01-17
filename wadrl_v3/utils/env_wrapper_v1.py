# wadrl/envs/pvrpd_env.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch

from utils.config_v2 import Config, euclid, precompute_drone_times, build_config_from_files


@dataclass
class StepResult:
    obs: Dict[str, torch.Tensor]
    reward: torch.Tensor
    done: torch.Tensor
    info: Dict


class PVRPDEnv:
    """
    Parallel VRPD Pick-up environment with:
      - weight-aware scalarization w1*F1 + w2*F2 (w1,w2>=0, w1+w2=1)
      - drone trip accounting: current trip index, optional max trips per drone
      - per-trip time cap and energy feasibility
      - two stepping modes:
          * serialized (vehicle gate): action=(veh_type, instance_idx, next_node)
          * parallel: action={'truck_nodes': Long[K], 'drone_nodes': Long[D]}
      - masks enforcing feasibility per head
    """

    def __init__(
        self,
        cfg: Config,
        *,
        keep_trip_rounds: bool = True,
        parallel: bool = False,
        add_idle: bool = False,
        max_trips_per_drone: Optional[int] = None,
        # reward normalization toggles (you can keep these False first):
        normalize_objectives: bool = False,
        # weight handling:
        weight_mode: str = "sample",  # "sample" (Dirichlet) or "fixed"
        fixed_weights: Tuple[float, float] = (0.5, 0.5),
        dirichlet_alpha: Tuple[float, float] = (1.0, 1.0),
        device: Optional[torch.device] = None,
        # max episode length
        max_steps: int | None = 512, timeout_penalty: float = -1e16,
        show_moves: bool = False,
        unserved_penalty = 1e9
    ):
        """
        Args:
          cfg: Config from your parser.
          keep_trip_rounds: if True, when drone returns to depot, E_cur resets and trip_idx += 1
          parallel: if True, step expects per-vehicle actions in one call
          add_idle: if True, add an idle token to masks (last column)
          max_trips_per_drone: optional hard cap on number of trips each drone may start
          normalize_objectives: if True, maintains running min/max for F1,F2 to normalize
          weight_mode: "sample" samples (w1,w2) ~ Dirichlet(alpha1,alpha2) at reset; "fixed" uses fixed_weights
          fixed_weights: weights used in inference if weight_mode="fixed"
          dirichlet_alpha: alpha parameters for sampling w1,w2 during training
        """

        self.show_moves = show_moves
        self.unserved_penalty = unserved_penalty

        self.max_steps = max_steps
        self.timeout_penalty = float(timeout_penalty)
        self.step_count = 0

        self.cfg = cfg
        self.keep_trip_rounds = keep_trip_rounds
        self.parallel = parallel
        self.add_idle = add_idle
        self.max_trips_per_drone = max_trips_per_drone
        self.normalize_objectives = normalize_objectives

        # weights
        self.weight_mode = weight_mode
        self.fixed_weights = tuple(float(x) for x in fixed_weights)
        self.dirichlet_alpha = np.asarray(dirichlet_alpha, dtype=np.float64)

        self.N = cfg.customers.N
        self.K = cfg.trucks.num_trucks
        self.D = cfg.drones.D
        self.device = torch.device(cfg.device) if device is None else device

        # geometry (index 0 is depot)
        self.coords = [(0.0, 0.0)] + [c.coord() for c in cfg.customers.items]
        self.demands = [0.0] + [c.demand for c in cfg.customers.items]
        self.only_truck = np.array([0] + [int(c.only_truck) for c in cfg.customers.items], dtype=np.int64)
        self.svc_trk = np.array([0.0] + [c.truck_service_time_s for c in cfg.customers.items], dtype=np.float32)
        self.svc_dr  = np.array([0.0] + [c.drone_service_time_s  for c in cfg.customers.items], dtype=np.float32)

        # precompute pairwise distance
        self.dist = self._pairwise_dist(self.coords)  # (V,V), float32 torch
        # drone time/energy matrices from catalog (D,V,V)
        T_dr = precompute_drone_times(cfg.drones, self.coords)
        self.T_dr = torch.tensor(T_dr, dtype=torch.float32, device=self.device)

        # per-trip time cap (optional) from your meta: cfg.drones.max_trip_time_s (may be None)
        # self.trip_time_cap = float(cfg.drones.max_trip_time_s) if cfg.drones.max_trip_time_s is not None else None
        self.trip_time_cap = None
        # running normalization for F1/F2 if enabled
        self._Fmin = torch.tensor([float('inf'), float('inf')], dtype=torch.float32, device=self.device)
        self._Fmax = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device)

        # idle token index (last column if enabled)
        self.idle_index = self.N + 1  # after nodes 0..N

        self._reset_state_buffers()
        self._reset_weights()

    # -------------------- public helpers --------------------

    def set_fixed_weights(self, w1: float, w2: float):
        """Use for inference: sets weight_mode='fixed' and stores (w1,w2) normalized."""
        s = float(w1) + float(w2)
        if s <= 0:
            raise ValueError("w1+w2 must be positive.")
        self.weight_mode = "fixed"
        self.fixed_weights = (float(w1) / s, float(w2) / s)

    def sample_weights(self):
        """Call to sample new (w1,w2) from Dirichlet (for training curricula)."""
        w = np.random.dirichlet(self.dirichlet_alpha)
        self.w = torch.tensor(w.astype(np.float32), device=self.device)  # (2,)
        return self.w

    # -------------------- core API --------------------

    def reset(self) -> Dict[str, torch.Tensor]:
        self.step_count = 0
        if self.max_steps is None:
            # heuristic: enough to visit all customers + returns, but bounded
            self.max_steps = 5 * (self.N + self.K + self.D)

        self.cargo_trk: List[List[int]] = [[] for _ in range(self.K)]
        self.cargo_dr: List[List[int]] = [[] for _ in range(self.D)]        
        self.cargo_mass_dr = torch.zeros(self.D, dtype=torch.float32, device=self.device)

        self.t_ret = torch.full((self.N+1,), float('nan'), device=self.device)
        self._reset_state_buffers()
        self._reset_weights()
        return self._build_obs()

    def step(self, action) -> StepResult:
        self.step_count += 1
        if self.step_count > self.max_steps and not self._check_done():
            print("Max episode len exceed!")
            reward = self.timeout_penalty
            obs = self._build_obs()
            info = self._info(True)
            return StepResult(
                obs=obs,
                reward=torch.tensor([reward], dtype=torch.float32, device=self.device),
                done=torch.tensor([True], dtype=torch.bool, device=self.device),
                info=info
            )
        if self.parallel:
            res = self._step_parallel(action)
        else:
            res = self._step_serialized(action)
        return res

    # -------------------- internals: state --------------------

    def _reset_state_buffers(self):
        V = self.N + 1

        # served flags
        self.served = torch.zeros(V, dtype=torch.bool, device=self.device)
        self.served[0] = True

        # vehicle clocks
        self.t_trk = torch.zeros(self.K, dtype=torch.float32, device=self.device)
        self.t_dr  = torch.zeros(self.D, dtype=torch.float32, device=self.device)

        # positions (start at depot)
        self.pos_trk = torch.zeros(self.K, dtype=torch.long, device=self.device)
        self.pos_dr  = torch.zeros(self.D, dtype=torch.long, device=self.device)

        # energy and trip bookkeeping (per drone)
        self.Ecur = torch.tensor([d.battery_power for d in self.cfg.drones.drones],
                                 dtype=torch.float32, device=self.device)
        self.trip_idx = torch.zeros(self.D, dtype=torch.long, device=self.device)  # current trip number
        self.trip_elapsed = torch.zeros(self.D, dtype=torch.float32, device=self.device)  # seconds spent within current trip

        # finished flags (optional if you want to close each tour)
        self.trk_done = torch.zeros(self.K, dtype=torch.bool, device=self.device)
        self.dr_done  = torch.zeros(self.D, dtype=torch.bool, device=self.device)

        self.cargo_dr: List[List[int]] = [[] for _ in range(self.D)]        
        self.cargo_mass_dr = torch.zeros(self.D, dtype=torch.float32, device=self.device)
        

        # accounting for objectives
        self.t_svc = torch.full((V,), float('nan'), dtype=torch.float32, device=self.device)  # service completion
        self.t_ret = torch.full((V,), float('nan'), dtype=torch.float32, device=self.device)  # return-to-depot times (optional exact)

    def _reset_weights(self):
        if self.weight_mode == "fixed":
            w1, w2 = self.fixed_weights
            s = max(1e-9, w1 + w2)
            self.w = torch.tensor([w1 / s, w2 / s], dtype=torch.float32, device=self.device)
        else:
            # sample (training)
            w = np.random.dirichlet(self.dirichlet_alpha)
            self.w = torch.tensor(w.astype(np.float32), device=self.device)

    def _drone_time(self, d: int, i: int, j: int) -> float:
        # dist = float(self.dist[i, j].item())
        # return float(self.cfg.drones.drones[d].flight_time_s(dist))
        return float(self.T_dr[d, i, j].item())

    def _drone_energy(self, d: int, i: int, j: int, payload_kg: float) -> float:
        dist = float(self.dist[i, j].item())
        return float(self.cfg.drones.drones[d].energy_j(dist, payload_kg))

    # -------------------- internals: utilities --------------------

    def _pairwise_dist(self, coords):
        V = len(coords)
        M = np.zeros((V, V), dtype=np.float32)
        for i in range(V):
            for j in range(V):
                if i != j:
                    M[i, j] = euclid(coords[i], coords[j])
        return torch.tensor(M, dtype=torch.float32, device=self.device)

    def _truck_travel_time(self, i: int, j: int, t0: float) -> float:
        return self.cfg.trucks.calc_truck_travel_time(start_time_s=t0, distance_m=float(self.dist[i, j].item()))

    # -------------------- observation & masks --------------------

    def _build_obs(self) -> Dict[str, torch.Tensor]:
        V = self.N + 1

        # Graph features: (x,y, only_truck, demand, svc_trk, svc_dr, served)
        fx = torch.empty((V, 7), dtype=torch.float32, device=self.device)
        fx[:, 0:2] = torch.tensor(self.coords, dtype=torch.float32, device=self.device)
        fx[:, 2]   = torch.tensor(self.only_truck, dtype=torch.float32, device=self.device)
        fx[:, 3]   = torch.tensor(self.demands,    dtype=torch.float32, device=self.device)
        fx[:, 4]   = torch.tensor(self.svc_trk,    dtype=torch.float32, device=self.device)
        fx[:, 5]   = torch.tensor(self.svc_dr,     dtype=torch.float32, device=self.device)
        fx[:, 6]   = self.served.float()
        graph_ctx = fx.unsqueeze(0)  # (1,V,7)

        # Trucks context: (x,y, available, slot_sin, slot_cos)
        K = self.K
        t_feats = torch.zeros((K, 5), dtype=torch.float32, device=self.device)
        t_feats[:, 0:2] = torch.tensor([self.coords[i] for i in self.pos_trk.tolist()],
                                       dtype=torch.float32, device=self.device)
        t_feats[:, 2] = (~self.trk_done).float()
        L = self.cfg.trucks.L
        slot_idx = torch.floor(self.t_trk / self.cfg.trucks.slot_len_seconds) % L
        ang = 2.0 * math.pi * slot_idx / L
        t_feats[:, 3] = torch.sin(ang)
        t_feats[:, 4] = torch.cos(ang)
        trucks_ctx = t_feats.unsqueeze(0)  # (1,K,5)

        # Drones context: (x,y, v_to, v_cru, v_ld, cap, E_max, E_cur, trip_idx, avail, trip_elapsed, trips_left)
        D = self.D
        d_feats = torch.zeros((D, 12), dtype=torch.float32, device=self.device)
        d_feats[:, 0:2] = torch.tensor([self.coords[i] for i in self.pos_dr.tolist()],
                                       dtype=torch.float32, device=self.device)
        trips_left = torch.full((D,), float('inf'), dtype=torch.float32, device=self.device)
        if self.max_trips_per_drone is not None:
            trips_left = (self.max_trips_per_drone - self.trip_idx).clamp(min=0).float()
        for dd, drone in enumerate(self.cfg.drones.drones):
            d_feats[dd, 2] = drone.takeoff_speed
            d_feats[dd, 3] = drone.cruise_speed
            d_feats[dd, 4] = drone.landing_speed
            d_feats[dd, 5] = drone.capacity
            d_feats[dd, 6] = drone.battery_power
        d_feats[:, 7]  = self.Ecur
        d_feats[:, 8]  = self.trip_idx.float()
        d_feats[:, 9]  = (~self.dr_done).float()
        d_feats[:, 10] = self.trip_elapsed
        d_feats[:, 11] = trips_left
        drones_ctx = d_feats.unsqueeze(0)  # (1,D,12)

        # masks
        mask_trk = self._mask_truck()    # (1,K,V [+1 if idle])
        mask_dr  = self._mask_drone()    # (1,D,V [+1 if idle])

        # weights (1,2) to be embedded by the policy (WADRL)
        weights = self.w.view(1, 2)

        return dict(
            graph_ctx=graph_ctx,
            trucks_ctx=trucks_ctx,
            drones_ctx=drones_ctx,
            mask_trk=mask_trk,
            mask_dr=mask_dr,
            weights=weights,
        )

    def _mask_truck(self):
        V = self.N + 1
        K = self.K

        extra = 1 if self.add_idle else 0
        mask = torch.zeros((1, K, V + extra), dtype=torch.bool, device=self.device)

        # feasibility: can visit any unserved customer, plus depot (always)
        # (You can add more truck-specific constraints here if needed)
        # customers 1..N:
        feas_cust = (~self.served[1:]).unsqueeze(0).unsqueeze(0).expand(1, K, self.N)  # (1,K,N)
        mask[:, :, 1:1+self.N] = feas_cust
        # depot always allowed
        mask[:, :, 0] = False

        if self.add_idle:
            mask[:, :, -1] = True  # idle

        # if a truck is marked done, only idle (if enabled) or depot (to keep it harmless)
        done_idx = torch.where(self.trk_done)[0].tolist()
        for k in done_idx:
            mask[:, k, :] = False
            mask[:, k, 0] = True
            if self.add_idle:
                mask[:, k, -1] = True

        for k in range(K):
            cur = int(self.pos_trk[k].item())
            mask[:, k, cur] = False          # forbid self-loop everywhere

        return mask

    def _mask_drone(self):
        V = self.N + 1
        D = self.D
        extra = 1 if self.add_idle else 0
        mask = torch.zeros((1, D, V + extra), dtype=torch.bool, device=self.device)

        # base feasibility: not truck-only and not served
        not_truck_only = torch.tensor(1 - self.only_truck, dtype=torch.bool, device=self.device)  # 1 for drone-allowed
        base_ok = (not_truck_only & (~self.served)).unsqueeze(0).unsqueeze(0).expand(1, D, V)  # (1,D,V)
        mask[:, :, :] = False
        mask[:, :, :] |= base_ok

        # depot always allowed
        mask[:, :, 0] = True

        # energy + return reachability + per-trip time cap + max_trips constraint
        for d in range(D):
            if self.dr_done[d]:
                mask[:, d, :] = False
                mask[:, d, 0] = True
                if self.add_idle:
                    mask[:, d, -1] = True
                continue

            i = int(self.pos_dr[d].item())
            Ecur = float(self.Ecur[d].item())
            trip_elapsed = float(self.trip_elapsed[d].item())

            # If max_trips_per_drone reached and drone is at depot, forbid leaving depot:
            if (self.max_trips_per_drone is not None) and (self.trip_idx[d] >= self.max_trips_per_drone) and (i == 0):
                mask[:, d, :] = False
                mask[:, d, 0] = True  # can sit at depot only
                if self.add_idle:
                    mask[:, d, -1] = True
                continue

            # Check each candidate j:
            for j in range(V):
                payload = float(self.cargo_mass_dr[d].item())
                cap = self.cfg.drones.drones[d].capacity
                Ecur = float(self.Ecur[d].item())

                if j == i:
                    mask[:, d, j] = False  # forbid self-loop
                    continue

                if j == 0:
                    # Depot return feasibility
                    e_i0 = self._drone_energy(d, i, 0, payload)
                    if Ecur < e_i0: 
                        mask[:, d, 0] = False
                        continue
                    if self.trip_time_cap is not None:
                        t_i0 = self._drone_time(d, i, 0)
                        if self.trip_elapsed[d] + t_i0 > self.trip_time_cap:
                            mask[:, d, 0] = False
                    continue

                # customer j
                if self.served[j] or self.only_truck[j]:
                    mask[:, d, j] = False
                    continue
                # capacity after pickup
                if payload + self.demands[j] > cap:
                    mask[:, d, j] = False
                    continue

                # energy forward + guaranteed return (after serving j)
                e_ij = self._drone_energy(d, i, j, payload)
                e_j0 = self._drone_energy(d, j, 0, payload + self.demands[j])
                if Ecur < e_ij + e_j0 + 1e-9:
                    mask[:, d, j] = False
                    continue

                # per-trip time cap (forward+svc+return)
                if self.trip_time_cap is not None:
                    t_ij = self._drone_time(d, i, j)
                    t_j0 = self._drone_time(d, j, 0)
                    svc  = float(self.svc_dr[j]) if not isinstance(self.svc_dr, torch.Tensor) else float(self.svc_dr[j].item())
                    if self.trip_elapsed[d] + t_ij + svc + t_j0 > self.trip_time_cap:
                        mask[:, d, j] = False
                        continue

                mask[:, d, j] = True

            if self.add_idle:
                mask[:, d, -1] = True  # idle

        # served customers masked:
        mask[:, :, 1:1+self.N] &= (~self.served[1:]).unsqueeze(0).unsqueeze(0)

        # truck-only customers masked:
        truck_only = torch.tensor(self.only_truck, dtype=torch.bool, device=self.device)
        mask[:, :, :] &= torch.logical_not(truck_only).unsqueeze(0).unsqueeze(0) | torch.tensor(
            [[True] + [False]*self.N], device=self.device
        ).unsqueeze(1)

        return mask

    # -------------------- stepping --------------------

    def _apply_truck_move(self, k: int, j: int):
        i = int(self.pos_trk[k].item())
        if i == j and i != 0:
            raise ValueError("Self loop tour! Start != End is must!")
        t0 = float(self.t_trk[k].item())

        # if idle token used:
        if self.add_idle and (j == self.idle_index):
            # do nothing, could add small time tick if desired
            return

        dt = self._truck_travel_time(i, j, t0)
        self.t_trk[k] = t0 + dt + float(self.svc_trk[j])
        self.pos_trk[k] = j

        if j != 0:
            self.served[j] = True
            self.t_svc[j] = self.t_trk[k]
            self.cargo_trk[k].append(j)
        elif i != 0:  # returning from a route end (or multi-tour)
            t_back = float(self.t_trk[k].item())
            for cust in self.cargo_trk[k]:
                self.t_ret[cust] = t_back
            self.cargo_trk[k].clear()
            self.trk_done[k] = True



    def _apply_drone_move(self, d: int, j: int):
        i = int(self.pos_dr[d].item())
        if i == j:
            raise ValueError("Self loop tour! Start != End is must!")

        payload = float(self.cargo_mass_dr[d].item())

        if self.add_idle and (j == self.idle_index):
            return

        if j == 0:
            # Return to depot with current payload
            t_i0 = self._drone_time(d, i, 0)
            e_i0 = self._drone_energy(d, i, 0, payload)
            # safety mirrors mask:
            if float(self.Ecur[d].item()) < e_i0:
                raise ValueError("Energy violation on return")
            if self.trip_time_cap is not None and (self.trip_elapsed[d] + t_i0 > self.trip_time_cap):
                raise ValueError("Trip time cap violation on return")

            self.t_dr[d] += t_i0
            self.trip_elapsed[d] += t_i0
            self.Ecur[d] -= e_i0
            self.pos_dr[d] = 0

            # finalize waiting times for all onboard goods
            t_back = float(self.t_dr[d].item())
            for cust in self.cargo_dr[d]:
                self.t_ret[cust] = t_back
            self.cargo_dr[d].clear()
            self.cargo_mass_dr[d] = 0.0

            if self.keep_trip_rounds:
                self.trip_idx[d] += 1
                self.trip_elapsed[d] = 0.0
                self.Ecur[d] = self.cfg.drones.drones[d].battery_power
            return

        # Move to customer j
        t_ij = self._drone_time(d, i, j)
        e_ij = self._drone_energy(d, i, j, payload)
        svc  = float(self.svc_dr[j]) if not isinstance(self.svc_dr, torch.Tensor) else float(self.svc_dr[j].item())

        # safety mirrors mask (including guaranteed return)
        e_j0 = self._drone_energy(d, j, 0, payload + self.demands[j])
        if float(self.Ecur[d].item()) < e_ij + e_j0:
            raise ValueError("Energy violation forward+return")
        if payload + self.demands[j] > self.cfg.drones.drones[d].capacity:
            raise ValueError("Capacity violation")
        if self.trip_time_cap is not None:
            t_j0 = self._drone_time(d, j, 0)
            if self.trip_elapsed[d] + t_ij + svc + t_j0 > self.trip_time_cap:
                raise ValueError("Trip time cap violation")

        # apply
        self.t_dr[d] += t_ij + svc
        self.trip_elapsed[d] += t_ij + svc
        self.Ecur[d] -= e_ij
        self.pos_dr[d] = j

        # mark service & load
        self.served[j] = True
        self.t_svc[j] = self.t_dr[d]
        self.cargo_dr[d].append(j)
        self.cargo_mass_dr[d] += float(self.demands[j])


    def _step_serialized(self, action) -> StepResult:
        """
        action: tuple/list (veh_type: 0=truck,1=drone, instance_idx, next_node_id)
                next_node_id in [0..N] or idle index if add_idle=True
        """
        veh_type, inst, j = map(int, action)
        if self.show_moves:
            print(f"Action: {veh_type} with index {inst} move to {j}")
        if veh_type == 0:
            self._apply_truck_move(inst, j)
        else:
            self._apply_drone_move(inst, j)

        done = self._check_done()
        if done and self.show_moves:
            print("Episode done!")
        reward = self._compute_step_reward(done)
        obs = self._build_obs()
        info = self._info(done)
        return StepResult(
            obs=obs,
            reward=torch.tensor([reward], dtype=torch.float32, device=self.device),
            done=torch.tensor([done], dtype=torch.bool, device=self.device),
            info=info
        )

    def _step_parallel(self, action) -> StepResult:
        """
        action: dict {'truck_nodes': LongTensor[K], 'drone_nodes': LongTensor[D]}
                each entry is next node id (0..N) or idle index (if enabled)
        """
        truck_nodes = action['truck_nodes'].tolist()
        drone_nodes = action['drone_nodes'].tolist()

        for k, j in enumerate(truck_nodes):
            self._apply_truck_move(k, int(j))
        for d, j in enumerate(drone_nodes):
            self._apply_drone_move(d, int(j))

        done = self._check_done()
        # if done:
        #     print("Episode end!")
        reward = self._compute_step_reward(done)
        obs = self._build_obs()
        info = self._info(done)
        return StepResult(
            obs=obs,
            reward=torch.tensor([reward], dtype=torch.float32, device=self.device),
            done=torch.tensor([done], dtype=torch.bool, device=self.device),
            info=info
        )

    # -------------------- termination, objectives, reward --------------------

    def _check_done(self) -> bool:
        # Here: episode ends when all customers are served.
        # (Optionally also require all vehicles to be back at depot.)
        return bool(self.served[1:].all().item())

    def _compute_F1_F2(self) -> Tuple[float, float]:
        # t_ret currently has NaNs for cargo still onboard; we fill those with
        # a conservative “return now” completion time per vehicle.
        t_ret = self.t_ret.clone()

        # Trucks: if carrying, estimate immediate return completion for those samples
        for k in range(self.K):
            if self.cargo_trk[k]:
                i = int(self.pos_trk[k].item())
                t0 = float(self.t_trk[k].item())
                dt_back = self._truck_travel_time(i, 0, t0)
                t_back = t0 + dt_back
                for cust in self.cargo_trk[k]:
                    t_ret[cust] = t_back

        # Drones: if carrying, estimate immediate return time using _drone_time
        for d in range(self.D):
            if self.cargo_dr[d]:
                i = int(self.pos_dr[d].item())
                t_back = float(self.t_dr[d].item()) + self._drone_time(d, i, 0)
                for cust in self.cargo_dr[d]:
                    t_ret[cust] = t_back


        # Valid served customers (exclude depot=0)
        served_mask = self.served.clone()
        served_mask[0] = False

        unserved_mask = ~self.served.clone()
        unserved_mask[0] = False
        num_sunsev = int(unserved_mask.sum().item())

        # if self.show_moves and not self._check_done():
        #     all_customers = torch.arange(self.N + 1, device=self.device)  # includes depot 0
        #     nonserved_idx = all_customers[~served_mask].tolist()
        #     nonserved_idx = [i for i in nonserved_idx if i != 0]
        #     served_missing_ret = all_customers[served_mask & torch.isnan(t_ret)].tolist()
        #     served_missing_svc = all_customers[served_mask & torch.isnan(self.t_svc)].tolist()
        #     if nonserved_idx:
        #         print(f"[DEBUG] Unserved customers: {nonserved_idx}")
        #     else:
        #         print("[DEBUG] All customers served.")
        #     if served_missing_ret:
        #         print(f"[DEBUG WARN] Served customers missing t_ret (NaN): {served_missing_ret}")
        #     if served_missing_svc:
        #         print(f"[DEBUG WARN] Served customers missing t_svc (NaN): {served_missing_svc}")
        #     carrying_trk = {k: list(self.cargo_trk[k]) for k in range(self.K) if len(self.cargo_trk[k]) > 0}
        #     carrying_dr  = {d: list(self.cargo_dr[d])  for d in range(self.D) if len(self.cargo_dr[d])  > 0}
        #     if carrying_trk:
        #         print(f"[DEBUG] Trucks carrying cargo (cust IDs): {carrying_trk}")
        #     if carrying_dr:
        #         print(f"[DEBUG] Drones carrying cargo (cust IDs): {carrying_dr}")


        # ---------- F1: makespan over served customers ----------
        # valid entries are those that are served and not NaN in t_ret
        valid_f1 = served_mask & (~torch.isnan(t_ret))
        if valid_f1.any():
            # replace invalid entries with a very negative number, then max
            very_neg = torch.tensor(-1e30, device=self.device, dtype=t_ret.dtype)
            t_ret_safe = torch.where(valid_f1, t_ret, very_neg)
            F1 = float(t_ret_safe.max().item())
        else:
            F1 = 0.0

        # ---------- F2: total waiting time sum_i (t_ret[i] - t_svc[i]) ----------
        wait = t_ret - self.t_svc
        valid_f2 = served_mask & (~torch.isnan(wait))
        F2 = float(torch.where(valid_f2, wait, torch.tensor(0.0, device=self.device)).sum().item())
        if num_sunsev > 0:
            F2 += num_sunsev * self.unserved_penalty
        return F1, F2

    def _compute_step_reward(self, done: bool) -> float:
        # episodic: only give reward when all customers served
        if not done:
            return 0.0
        F1, F2 = self._compute_F1_F2()
        w1, w2 = float(self.w[0].item()), float(self.w[1].item())
        return -(w1 * F1 + w2 * F2)


    def _info(self, done: bool) -> Dict:
        F1, F2 = self._compute_F1_F2()
        reward = self._compute_step_reward(self._check_done())
        return dict(
            reward=reward,
            F1=F1, F2=F2,
            w1=float(self.w[0].item()), w2=float(self.w[1].item()),
            served=int(self.served[1:].sum().item()),
        )

if __name__ == "__main__":
    # build cfg with your parser
# cfg, meta = build_config_from_files(...)
    cfg, meta = build_config_from_files(
        truck_json_path="/home/saphyiera/HUST/WADRL_PVRP/data/Truck_config.json",
        drone_json_path="/home/saphyiera/HUST/WADRL_PVRP/data/drone_linear_config.json",
        customers_txt_path="/home/saphyiera/HUST/WADRL_PVRP/data/random_data/6.5.1.txt")
    env = PVRPDEnv(
        cfg,
        keep_trip_rounds=True,
        parallel=False,        # or True if you want per-vehicle decisions each step
        add_idle=False,        # set True for parallel+idle
        max_trips_per_drone=5, # optional hard cap
        normalize_objectives=False,
        weight_mode="sample",  # training
        dirichlet_alpha=(1.0, 1.0),
    )

    obs = env.reset()              # obs includes 'weights' (1,2)
    # Serialized action example:
    #   veh_type=0 (truck), instance=0..K-1, next_node in [0..N] (or env.idle_index if add_idle)
    step = env.step((0, 0, 1))
    step = env.step((0, 0, 2))
    step = env.step((0, 0, 3))
    step = env.step((0, 0, 4))
    step = env.step((0, 0, 5))
    step = env.step((0, 0, 6))
    step = env.step((0, 0, 0))
    print(f"Step Reward: {step.reward}, F1: {step.info['F1']}, F2: {step.info['F2']}, "
      f"W1: {step.info['w1']}, W2: {step.info['w2']}")

    # Inference with fixed weights:
    env.set_fixed_weights(0.2, 0.8)  # minimize waiting time more
    obs = env.reset()
