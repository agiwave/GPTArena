
import aka.nn as nn
import aka.numpy as np
from MLP import MLPBlock

def MoeBlock(**kwargs):
    def forward(self, inputs):
        gate_logits = self.gate(inputs)
        weights, selected_experts = np.topk(gate_logits, self.num_experts_per_tok)
        weights = np.softmax(weights, dim=1, dtype=np.float).to(inputs.dtype)
        results = np.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = np.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results

    return nn.Module(
        experts = [MLPBlock(args) for _ in range(kwargs['num_experts'])],
        gate = nn.Linear(kwargs['latent_dim'], kwargs['num_experts'], bias=False),
        num_experts_per_tok = kwargs['num_experts_per_tok']
    )
