"""
Design 6 — Generate-and-Verify (GAV)
Breaks assumption A5: reporting is verification (inverse inference), not generation.

Smoke proxy: train a CLIP-style CT-text aligner on subset_5k, then at inference
sample K candidates from baseline Reg2RG and pick the highest-scoring one.
Training phase: learn the aligner via contrastive loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .base_novel import NovelArchitectureBase


class CTTextAligner(nn.Module):
    """CLIP-style cross-modal aligner for CT volumes and report text."""

    def __init__(self, vis_dim=768, text_dim=4096, proj_dim=512, temperature=0.07):
        super().__init__()
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(temperature).log())

    def forward(self, vis_feats, text_feats):
        """
        Args:
            vis_feats: (B, D_vis) pooled visual features
            text_feats: (B, D_text) pooled text features
        Returns:
            logits_per_image: (B, B) similarity matrix
        """
        v = F.normalize(self.vis_proj(vis_feats), dim=-1)
        t = F.normalize(self.text_proj(text_feats), dim=-1)
        temp = self.temperature.exp()
        logits = v @ t.T / temp
        return logits


class GenerateVerifyModel(NovelArchitectureBase):
    """GAV: train aligner, rerank candidate reports at inference."""

    def __init__(self, base_model, design_config):
        super().__init__(base_model, design_config)
        self.aligner = CTTextAligner()
        self.num_candidates = design_config.get("extra", {}).get("num_candidates", 10)

    def _get_text_features(self, input_embedding, attention_mask):
        """Extract pooled text features from LLM hidden states."""
        with torch.no_grad():
            out = self.base_model.lang_model(
                inputs_embeds=input_embedding,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1]  # (B, T, 4096)
            # Mean-pool over non-padding positions
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden)
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled

    def forward(self, lang_x, vision_x, mask_x, region2area, attention_mask, labels,
                gt_abnormality_labels=None, gt_severity_labels=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Visual features (pooled)
            vis_feats = self._extract_visual_features(vision_x)
            vis_pooled = vis_feats.mean(dim=1) if vis_feats is not None else None

            # Text features from GT report embeddings.
            # NOTE: my_embedding_layer.forward mutates BOTH vision_x AND mask_x
            # (overwrites values with embeddings). vision_x is safely shallow-
            # copied below; mask_x must be too, or TTA's per-iter re-call
            # crashes on iter 2 (mask_x[k] becomes 2D embedding, ViT3D wants 5D).
            vision_x_copy = dict(vision_x)
            mask_x_copy = dict(mask_x)
            input_embedding = self._get_standard_embeddings(vision_x_copy, mask_x_copy, lang_x, region2area)
            text_pooled = self._get_text_features(input_embedding, attention_mask)

            # Contrastive loss (InfoNCE)
            contrastive_loss = torch.tensor(0.0, device=lang_x.device)
            if vis_pooled is not None:
                logits = self.aligner(vis_pooled.float(), text_pooled.float())
                B = logits.shape[0]
                target = torch.arange(B, device=logits.device)
                loss_i2t = F.cross_entropy(logits, target)
                loss_t2i = F.cross_entropy(logits.T, target)
                contrastive_loss = (loss_i2t + loss_t2i) / 2

            # Also train LM for candidate generation
            output = self._run_llm(input_embedding, attention_mask, labels)
            total_loss = output['loss'] + contrastive_loss
            accuracy = self._compute_accuracy(output['logits'], labels)

        return {"loss": total_loss, "logits": accuracy, "contrastive_loss": contrastive_loss.detach()}

    def generate(self, lang_x, vision_x, mask_x, region2area, **kwargs):
        with torch.no_grad():
            # 2026-05-15: previously this function returned a 1-element list
            # regardless of batch size — it only kept `text[0]` of the batch-
            # decoded candidates and built vis_pooled / text_pooled with shape
            # (1, D). That silently dropped (bs-1) of every batch's samples,
            # corrupting all n06 cross-tier eval results. Now: loop over the
            # batch dim, return a list of length bs.
            # Shallow-copy BOTH vision_x and mask_x: my_embedding_layer.forward
            # mutates both (overwrites values with computed embeddings). TTA's
            # multi-pass loop must see fresh dicts on each iteration.
            vision_x_copy = dict(vision_x)
            mask_x_copy = dict(mask_x)
            input_embedding = self._get_standard_embeddings(vision_x_copy, mask_x_copy, lang_x, region2area)
            bs = input_embedding.shape[0]

            # K is overridable via env var N06_K (default = config or 5)
            import os as _os
            K = int(_os.environ.get("N06_K", min(self.num_candidates, 5)))

            # candidates_per_sample[i][k] = k-th candidate text for sample i
            candidates_per_sample = [[] for _ in range(bs)]
            for k in range(K):
                # Temperature schedule:
                #   default: 0.7 to 1.3 (broad sweep, original)
                #   N06_TEMP_LOW=1: 0.3 to 0.7 (more conservative)
                #   N06_TEMP_MID=1: 0.5 to 1.0 (added 2026-05-26 for sampler ablation)
                #   N06_TEMP_HIGH=1: 1.0 to 1.5 (added 2026-05-26 for sampler ablation)
                if _os.environ.get("N06_TEMP_LOW", "0") == "1":
                    t = 0.3 + (0.4 * k / max(K - 1, 1))
                elif _os.environ.get("N06_TEMP_MID", "0") == "1":
                    t = 0.5 + (0.5 * k / max(K - 1, 1))
                elif _os.environ.get("N06_TEMP_HIGH", "0") == "1":
                    t = 1.0 + (0.5 * k / max(K - 1, 1))
                else:
                    t = 0.7 + (0.6 * k / max(K - 1, 1))
                gen = self.base_model.lang_model.generate(
                    inputs_embeds=input_embedding,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=t,
                    top_k=50,
                )
                texts = self.base_model.text_tokenizer.batch_decode(gen, skip_special_tokens=True)
                for i in range(bs):
                    candidates_per_sample[i].append(texts[i] if i < len(texts) else "")

            # SCVS — Self-Consistency Voting Selection (added 2026-05-17).
            # Picks the candidate(s) that AGREE most with each other (high mean
            # Jaccard word-set similarity vs the other K-1 candidates), rather
            # than the verifier's best-pick or UNION's concat-all.
            #
            # Modes (gated by env vars):
            #   N06_SCVS=1                       -> return single most-self-consistent candidate per sample.
            #                                       Tests whether "consensus" beats "verifier max-score" in-dist.
            #   N06_SCVS=1 + N06_SCVS_TOPN=N>1   -> UNION-concat the top-N most-self-consistent candidates.
            #                                       Hybrid: keeps UNION's recall benefit while filtering out
            #                                       outlier (low-agreement) candidates.
            # No extra model, no extra VRAM — pure text-side set operations on K short strings.
            if _os.environ.get("N06_SCVS", "0") == "1":
                topn = int(_os.environ.get("N06_SCVS_TOPN", "1"))
                results = []
                for i in range(bs):
                    cands = candidates_per_sample[i]
                    if not cands:
                        results.append("")
                        continue
                    n = len(cands)
                    if n == 1:
                        results.append(cands[0])
                        continue
                    if topn >= n:
                        # asked for all -> UNION-concat all (equivalent to N06_UNION=1)
                        results.append(" ".join(cands))
                        continue
                    # Word-set per candidate (lowercased, whitespace-tokenized)
                    word_sets = [set(c.lower().split()) for c in cands]
                    # Mean Jaccard similarity to the other n-1 candidates, per candidate
                    mean_sims = []
                    for j in range(n):
                        sims = []
                        for kk in range(n):
                            if kk == j:
                                continue
                            a, b = word_sets[j], word_sets[kk]
                            union_sz = len(a | b)
                            sims.append(len(a & b) / union_sz if union_sz > 0 else 0.0)
                        mean_sims.append(sum(sims) / len(sims) if sims else 0.0)
                    # Top-N candidates by descending mean self-similarity
                    order = sorted(range(n), key=lambda x: mean_sims[x], reverse=True)
                    top_idx = order[:topn]
                    if topn == 1:
                        results.append(cands[top_idx[0]])
                    else:
                        results.append(" ".join(cands[ix] for ix in top_idx))
                return results

            # Cross-tier generalization mode: concatenate all K candidates per
            # sample instead of picking best. Hypothesis: the aligner verifier
            # picks one candidate that misses some conditions; union across all
            # K candidates increases recall, helping cross-tier F1 where the
            # GT report (CT-Rate impression) covers many findings.
            # Trigger via N06_UNION=1 env var.
            if _os.environ.get("N06_UNION", "0") == "1":
                return [" ".join(cands) for cands in candidates_per_sample]

            if K <= 1:
                return [cands[0] if cands else "" for cands in candidates_per_sample]

            # Score candidates with aligner — vectorize over batch dim.
            vis_feats = self._extract_visual_features(vision_x)
            if vis_feats is None:
                # Fall back to first candidate per sample
                return [cands[0] for cands in candidates_per_sample]
            vis_pooled = vis_feats.mean(dim=1)  # (bs, D)
            # Defensive: ensure bs matches; if not, fall back to first candidate.
            if vis_pooled.shape[0] != bs:
                return [cands[0] for cands in candidates_per_sample]

            best_reports = []
            for i in range(bs):
                best_score = float('-inf')
                best_report = candidates_per_sample[i][0]
                vis_i = vis_pooled[i:i+1].float()  # (1, D)
                for cand in candidates_per_sample[i]:
                    tokens = self.base_model.text_tokenizer(
                        cand, return_tensors="pt", truncation=True, max_length=512, padding=True
                    ).to(vis_i.device)
                    text_embeds = self.base_model.lang_model.get_input_embeddings()(tokens.input_ids)
                    text_pooled = text_embeds.mean(dim=1)  # (1, D)
                    logit = self.aligner(vis_i, text_pooled.float())[0, 0]
                    if logit > best_score:
                        best_score = logit
                        best_report = cand
                best_reports.append(best_report)

            return best_reports
