import torch
import torch.nn as nn


class MoE(nn.Module):
    def __init__(
        self,
        input_dim,
        experts=2,
        expert_hidden=128,
        gate_hidden=128,
        dropout=0.25,
        input_dropout=0.05,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.experts = nn.ModuleList( #expert network
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden),
                    nn.LayerNorm(expert_hidden),
                    nn.GELU(), #gaussian 
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, 1),
                )
                for _ in range(experts)
            ]
        )
        # gate to decide weight for each expert
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, experts),
        )

    def forward(self, x):
        x = self.input_dropout(x.float())
        gate_logits = self.gate(x)
        gate_weights = torch.softmax(gate_logits, dim=1)
        expert_logits = torch.cat([expert(x) for expert in self.experts], dim=1) #combine experts using gate weight
        logits = (expert_logits * gate_weights).sum(dim=1, keepdim=True) # weighted average
        return logits, gate_weights, expert_logits

# inference
def infer_config_from_state_dict(state_dict): #get weights from dict
    gate_weight = state_dict.get("gate.0.weight")
    if gate_weight is None:
        raise ValueError("missing 'gate.0.weight'.")

    input_dim = int(gate_weight.shape[1])
    gate_hidden = int(gate_weight.shape[0])

    expert_ids = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("experts.") and key.endswith(".0.weight")
        }
    )
    if not expert_ids:
        raise ValueError("no expert weights.")

    first_expert = f"experts.{expert_ids[0]}.0.weight"
    expert_hidden = int(state_dict[first_expert].shape[0])

    return {
        "input_dim": input_dim,
        "experts": len(expert_ids),
        "expert_hidden": expert_hidden,
        "gate_hidden": gate_hidden,
        "dropout": 0.0,
        "input_dropout": 0.0, # dropout not used during inference, just set to 0.0
    }
