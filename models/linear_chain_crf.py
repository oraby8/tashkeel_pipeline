import torch
import torch.nn as nn

class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int = 15):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1)

        numerator = self._compute_gold_score(emissions, tags, mask)
        denominator = self._compute_partition_function(emissions, mask)
        ll = numerator - denominator
        
        num_tokens = mask.float().sum()
        return -ll.sum() / max(num_tokens, 1.0)

    def _compute_gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = emissions.shape
        score = self.start_transitions[tags[0]] + emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_len):
            mask_i = mask[i]
            prev_tags = tags[i - 1]
            curr_tags = tags[i]

            emission_score = emissions[i, torch.arange(batch_size), curr_tags]
            transition_score = self.transitions[prev_tags, curr_tags]

            step_score = emission_score + transition_score
            score = score + step_score * mask_i.float()

        last_tags = self._get_last_tags(tags, mask)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_partition_function(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[0]

        for i in range(1, seq_len):
            mask_i = mask[i].unsqueeze(1)
            broadcast_alpha = alpha.unsqueeze(2)
            broadcast_trans = self.transitions.unsqueeze(0)
            broadcast_emiss = emissions[i].unsqueeze(1)

            next_alpha = torch.logsumexp(broadcast_alpha + broadcast_trans + broadcast_emiss, dim=1)
            alpha = torch.where(mask_i, next_alpha, alpha)

        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def _get_last_tags(self, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size = tags.shape
        lengths = mask.long().sum(dim=0) - 1
        lengths = torch.clamp(lengths, min=0)
        return tags[lengths, torch.arange(batch_size)]

    @torch.inference_mode()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list:
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)
        seq_len, batch_size, _ = emissions.shape

        viterbi_vars = self.start_transitions.unsqueeze(0) + emissions[0]
        backpointers = []

        for i in range(1, seq_len):
            mask_i = mask[i].unsqueeze(1)
            broadcast_viterbi = viterbi_vars.unsqueeze(2)
            broadcast_trans = self.transitions.unsqueeze(0)

            scores = broadcast_viterbi + broadcast_trans
            best_score, best_idx = torch.max(scores, dim=1)

            next_viterbi = best_score + emissions[i]
            viterbi_vars = torch.where(mask_i, next_viterbi, viterbi_vars)
            backpointers.append(best_idx)

        viterbi_vars = viterbi_vars + self.end_transitions.unsqueeze(0)
        best_last_tags = torch.argmax(viterbi_vars, dim=1)

        best_paths = []
        lengths = mask.long().sum(dim=0).cpu().tolist()

        for b in range(batch_size):
            L = lengths[b]
            if L == 0:
                best_paths.append([])
                continue
            best_tag = best_last_tags[b].item()
            best_path = [best_tag]
            for step in range(L - 2, -1, -1):
                best_tag = backpointers[step][b, best_tag].item()
                best_path.append(best_tag)
            best_path.reverse()
            best_paths.append(best_path)

        return best_paths
