import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedQMixer(nn.Module):
    def __init__(self, args):
        super(MaskedQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.mask_prob = getattr(args, "mask_prob", 0.0)
        self.is_sticky = args.is_sticky
        self.is_fixed = args.is_fixed
        self.no_state = args.no_state
        self.local_obs_instead_of_state = args.local_obs_instead_of_state

        print(f"state_dim: {self.state_dim}")  # #########
        print(50*"-")
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        bs, ep_len, _ = agent_qs.shape
        states = states.reshape(-1, self.state_dim)
        # Generate a random mask with probability mask_prob
        if self.mask_prob > 0.0 and self.training:
            if self.is_fixed:
                # Fixed-k masking: mask exactly k agents (based on mask_prob)
                mask_k = max(1, int(round(self.mask_prob * self.n_agents)))
                if self.is_sticky:
                    episode_mask = th.ones(bs, self.n_agents, device=agent_qs.device)
                    for b in range(bs):
                        idx = th.randperm(self.n_agents)[:mask_k]
                        episode_mask[b, idx] = 0.0
                    mask = episode_mask.unsqueeze(1).expand(-1, ep_len, -1)
                else:
                    mask = th.ones_like(agent_qs)
                    for b in range(bs):
                        for t in range(ep_len):
                            idx = th.randperm(self.n_agents)[:mask_k]
                            mask[b, t, idx] = 0.0
            else:
                # Probabilistic masking: mask each agent independently with mask_prob
                if self.is_sticky:
                    episode_mask = (
                        th.rand(bs, self.n_agents, device=agent_qs.device) > self.mask_prob
                    ).float()
                    mask = episode_mask.unsqueeze(1).expand(-1, ep_len, -1)
                else:
                    mask = (
                        th.rand(agent_qs.shape, device=agent_qs.device) > self.mask_prob
                    ).float()
            # print(f"is sticky {self.is_sticky}")
            # print(f"is fixed {self.is_fixed}")
            # print(f"mask: {mask}")
            agent_qs = agent_qs * mask

        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # Using the same mask for state masking
        flat_mask = mask.view(-1, self.n_agents)  # shape: (bs * ep_len, n_agents)

        if self.local_obs_instead_of_state:
            print(f"agent 0 is masked? :{flat_mask[0][0] == 0}")
            # states.shape: bs * ep_len, state_dim
            print(f"Unmasked state for agt 0 at t=0: {states[0][0]}")
            obs_dim = self.state_dim // self.n_agents
            # Reshape state: (bs * ep_len, n_agents, obs_dim)
            states = states.view(-1, self.n_agents, obs_dim)
            # Apply mask: (bs * ep_len, n_agents, 1)
            states = states * flat_mask.unsqueeze(-1)
            # Flatten back to (bs * ep_len, state_dim)
            states = states.view(-1, self.state_dim)
            print(f"Masked state for agt 0 at t=0: {states[0][0]}")

        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
