import math
import torch
from .various_divergence import VariousDivergence
from .soft_dtw_cuda import SoftDTW


class DualSpaceKDWithCMA(VariousDivergence):
    def __init__(self, config, distiller, padding_id=-100) -> None:
        super().__init__(config, padding_id=padding_id)
        self.dtw_rate = config.dtw_rate
        if self.dtw_rate > 0:
            self.dtw = SoftDTW(use_cuda=True, gamma=config.dtw_gamma)
        self.dtw_gamma_start = config.dtw_gamma_start
        self.dtw_gamma_end = config.dtw_gamma_end
        self.dtw_gamma_steps = config.dtw_gamma_steps
        self.dtw_band_width = config.dtw_band_width
        self.dtw_band_penalty = config.dtw_band_penalty
        self.dtw_band_center_blend = config.dtw_band_center_blend
        self.dtw_band_entropy_coef = config.dtw_band_entropy_coef
        self.dtw_band_warmup_steps = config.dtw_band_warmup_steps
        self._global_step = 0
        self.kd_warmup_steps = config.kd_warmup_steps
        self.dtw_warmup_steps = config.dtw_warmup_steps
        self.dtw_band_source = config.dtw_band_source
        self.distiller = distiller

    def compute_t2s_logits(self, concatenated_batch, distiller, model, reference_model):
        import torch, math

        # === Xác định device đang chạy ===
        device = next(model.parameters()).device  

        # Đảm bảo model và teacher nằm đúng device
        model = model.to(device)
        teacher_model = reference_model.to(device)
        teacher_model.eval()

        # === Chỉ chuyển projector nhỏ sang GPU ===
        for name in ["query", "t2s"]:
            if name in distiller.projectors:
                distiller.projectors[name] = distiller.projectors[name].to(device)

        # === Đưa batch sang device (gọn, không thiếu key nào) ===
        def move_to_device(obj):
            if isinstance(obj, dict):
                return {k: move_to_device(v) for k, v in obj.items()}
            elif torch.is_tensor(obj):
                return obj.to(device)
            else:
                return obj

        concatenated_batch = move_to_device(concatenated_batch)

        # === Forward teacher ===
        teacher_outputs = teacher_model(
            concatenated_batch["concatenated_teacher_input_ids"],
            attention_mask=concatenated_batch["concatenated_teacher_attention_mask"],
            output_hidden_states=True,
        )

        target = concatenated_batch["concatenated_student_labels"]
        teacher_target = concatenated_batch["concatenated_teacher_labels"]

        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        # === Lấy embedding layer student ===
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            stu_embed_tokens = model.model.embed_tokens
        elif hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "embed_tokens"):
            stu_embed_tokens = model.model.model.embed_tokens
        elif hasattr(model, "transformer") and hasattr(model.transformer, "word_embeddings"):
            stu_embed_tokens = model.transformer.word_embeddings
        else:
            raise NotImplementedError("Không tìm thấy embedding layer cho student model")

        # === Lấy embedding layer teacher ===
        if hasattr(teacher_model, "model") and hasattr(teacher_model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.embed_tokens
        elif hasattr(teacher_model, "model") and hasattr(teacher_model.model, "model") and hasattr(teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.model.embed_tokens
        elif hasattr(teacher_model, "transformer") and hasattr(teacher_model.transformer, "wte"):
            tea_embed_tokens = teacher_model.transformer.wte
        else:
            raise NotImplementedError("Không tìm thấy embedding layer cho teacher model")

        # === Xử lý input & target embeddings ===
        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(pad_mask, concatenated_batch["concatenated_student_input_ids"], torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(teacher_pad_mask, concatenated_batch["concatenated_teacher_input_ids"], torch.zeros_like(teacher_target))
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        # === Ghép embedding theo logic gốc ===
        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        # === Chuẩn hóa các tensor teacher ===
        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        # === Projector ===
        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()
        tea_v_hiddens = distiller.projectors["t2s"](
            norm_teacher_hiddens + norm_tea_target_embeds
        ).float()

        # === Alignment & attention ===
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])

        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        t2s_weight = torch.softmax(align, -1)

        # === Output hidden + logits ===
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)

        lm_head_w = model.lm_head.weight.detach()

        t2s_logits = t2s_hiddens.matmul(lm_head_w.transpose(-1, -2))

        return t2s_logits



    def compute_dtw_loss(self, batch, distiller, model, reference_model):
        # === Xác định device đang chạy ===
        device = next(model.parameters()).device  
        model = model.to(device)
        teacher_model = reference_model.to(device)
        teacher_model.eval()

        # === Forward student ===
        outputs = model(
            batch["chosen_student_input_ids"].to(device),
            attention_mask=batch["chosen_student_attention_mask"].to(device),
            output_hidden_states=True
        )

        # === Forward teacher ===
        teacher_outputs = teacher_model(
            batch["chosen_teacher_input_ids"].to(device),
            attention_mask=batch["chosen_teacher_attention_mask"].to(device),
            output_hidden_states=True
        )

        # === Labels & masks ===
        target = batch["chosen_student_labels"].to(device)
        teacher_target = batch["chosen_teacher_labels"].to(device)

        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        # === Target embeddings (đảm bảo output cùng device) ===
        stu_target_embeds, tea_target_embeds = self._get_target_embeddings(
            distiller, batch, pad_mask, teacher_pad_mask, model, teacher_model
        )
        stu_target_embeds = stu_target_embeds.to(device)
        tea_target_embeds = tea_target_embeds.to(device)

        # === Hidden states ===
        hiddens = outputs.hidden_states[-1].to(device)
        teacher_hiddens = teacher_outputs.hidden_states[-1].to(device)

        # === Đưa projectors về cùng device ===
        for name in ["dtw_embed_t2s", "t2s"]:
            if name in distiller.projectors:
                distiller.projectors[name] = distiller.projectors[name].to(device)

        # === Projected teacher embeddings & loss ===
        projected_teacher_embeds = distiller.projectors["dtw_embed_t2s"](tea_target_embeds)
        loss_embed = self._calculate_alignment_loss(stu_target_embeds, projected_teacher_embeds, pad_mask, teacher_pad_mask)

        # === Projected teacher hidden & loss ===
        projected_teacher_hiddens = distiller.projectors["t2s"](teacher_hiddens)
        loss_hidden = self._calculate_alignment_loss(hiddens, projected_teacher_hiddens, pad_mask, teacher_pad_mask)

        # === Tổng DTW loss ===
        total_dtw_loss = loss_hidden + loss_embed

        return total_dtw_loss

    
    def _calculate_alignment_loss(self, student_embs, teacher_embs, student_mask, teacher_mask):
        batch_size = student_embs.size(0)
        total_loss = torch.tensor(0.0, device=student_embs.device, requires_grad=True)
        non_empty_pairs = 0

        for i in range(batch_size):
            s_len = student_mask[i].sum().item()
            t_len = teacher_mask[i].sum().item()

            if s_len == 0 or t_len == 0:
                continue
            
            non_empty_pairs += 1

            s_seq = student_embs[i, :s_len, :]
            t_seq = teacher_embs[i, :t_len, :]

            print("[DEBUG] s_seq shape:", s_seq.shape)
            print("[DEBUG] t_seq shape:", t_seq.shape)
            c_stu_tea = 1.0 - torch.cosine_similarity(
                s_seq.unsqueeze(1), t_seq.unsqueeze(0), dim=-1
            )

            c_stu_stu = 1.0 - torch.cosine_similarity(
                s_seq.unsqueeze(1), s_seq.unsqueeze(0), dim=-1
            )

            c_tea_tea = 1.0 - torch.cosine_similarity(
                t_seq.unsqueeze(1), t_seq.unsqueeze(0), dim=-1
            )
            
            if self.dtw_band_source == 'cma' and hasattr(self, 'last_align') and self.last_align is not None and self.dtw_band_width > 0:
                # last_align is (B, S, T). Slice i-th example and valid spans
                A = self.last_align[i][:s_len, :t_len]
                eps = 1e-9
                A_clamped = (A + eps) / (A.sum(dim=-1, keepdim=True) + eps)
                row_entropy = -(A_clamped * torch.log(A_clamped)).sum(dim=-1)  # (s_len)

                # Normalized teacher length mapping: i -> i * (t_len / s_len)
                lin_center = torch.arange(s_len, device=A.device, dtype=torch.float32) * (float(t_len) / float(s_len))
                soft_center = (A_clamped * torch.arange(t_len, device=A.device).view(1, -1)).sum(dim=-1)
                alpha = float(self.dtw_band_center_blend)
                centers = alpha * soft_center + (1.0 - alpha) * lin_center

                # Adaptive width per token
                base_w = float(self.dtw_band_width)
                width = base_w + float(self.dtw_band_entropy_coef) * row_entropy

                # Soft penalty mask
                j = torch.arange(t_len, device=A.device).view(1, -1).float()
                dist = (j - centers.view(-1, 1)).abs()
                band = dist <= width.view(-1, 1)

                # Warmup for penalty strength
                if self.dtw_band_warmup_steps and self.dtw_band_warmup_steps > 0:
                    pen_scale = min(1.0, float(self._global_step + 1) / float(self.dtw_band_warmup_steps))
                else:
                    pen_scale = 1.0
                penalty = float(self.dtw_band_penalty) * pen_scale
                c_stu_tea = c_stu_tea + (~band).float() * penalty

            if self.dtw_band_source == 'sdtw' and self.dtw_band_width > 0:
                _, A = self.dtw.forward_with_cost_matrix(c_stu_tea.unsqueeze(0), return_alignment=True)
                A = A[0]
                eps = 1e-9
                A_clamped = (A + eps) / (A.sum(dim=-1, keepdim=True) + eps)
                row_entropy = -(A_clamped * torch.log(A_clamped + eps)).sum(dim=-1)
                lin_center = torch.arange(s_len, device=A.device, dtype=torch.float32) * (float(t_len) / float(s_len))
                soft_center = (A_clamped * torch.arange(t_len, device=A.device).view(1, -1)).sum(dim=-1)
                alpha = float(self.dtw_band_center_blend)
                centers = alpha * soft_center + (1.0 - alpha) * lin_center
                base_w = float(self.dtw_band_width)
                width = base_w + float(self.dtw_band_entropy_coef) * row_entropy
                j = torch.arange(t_len, device=A.device).view(1, -1).float()
                dist = (j - centers.view(-1, 1)).abs()
                band = dist <= width.view(-1, 1)
                if self.dtw_band_warmup_steps and self.dtw_band_warmup_steps > 0:
                    pen_scale = min(1.0, float(self._global_step + 1) / float(self.dtw_band_warmup_steps))
                else:
                    pen_scale = 1.0
                penalty = float(self.dtw_band_penalty) * pen_scale
                c_stu_tea = c_stu_tea + (~band).float() * penalty

            s2t = self.dtw.forward_with_cost_matrix(c_stu_tea.unsqueeze(0))
            s2s = self.dtw.forward_with_cost_matrix(c_stu_stu.unsqueeze(0))
            t2t = self.dtw.forward_with_cost_matrix(c_tea_tea.unsqueeze(0))

            pair_loss = s2t - 0.5 * (s2s + t2t)
        
            total_loss = total_loss + pair_loss.view(1)


        if non_empty_pairs == 0:
            return torch.tensor(0.0, device=student_embs.device, requires_grad=True)

        return total_loss 
    
    '''
    def _calculate_alignment_loss(self, student_embs, teacher_embs, student_mask, teacher_mask):
        batch_size = student_embs.size(0)
        device = student_embs.device
        total_loss = torch.zeros(1, device=device, requires_grad=True)
        non_empty_pairs = 0

        def chunked_cosine_similarity(a, b, chunk_size=64):
            # a: (s_len, hidden), b: (t_len, hidden)
            s_len, hidden = a.shape
            t_len = b.shape[0]
            sim = torch.zeros(s_len, t_len, device=a.device)
            for i_start in range(0, s_len, chunk_size):
                i_end = min(i_start + chunk_size, s_len)
                a_chunk = a[i_start:i_end].unsqueeze(1)  # (chunk, 1, hidden)
                sim[i_start:i_end] = 1.0 - torch.cosine_similarity(
                    a_chunk, b.unsqueeze(0), dim=-1
                )
            return sim

        for i in range(batch_size):
            s_len = student_mask[i].sum().item()
            t_len = teacher_mask[i].sum().item()

            if s_len == 0 or t_len == 0:
                continue
            non_empty_pairs += 1

            s_seq = student_embs[i, :s_len, :]
            t_seq = teacher_embs[i, :t_len, :]

            # Sử dụng chunked cosine similarity để giảm memory
            c_stu_tea = chunked_cosine_similarity(s_seq, t_seq)
            c_stu_stu = chunked_cosine_similarity(s_seq, s_seq)
            c_tea_tea = chunked_cosine_similarity(t_seq, t_seq)

            # --- (Giữ nguyên phần xử lý dtw_band_source như cũ) ---
            # ... giữ nguyên logic CMA/SDTW band và penalty ...

            s2t = self.dtw.forward_with_cost_matrix(c_stu_tea.unsqueeze(0))
            s2s = self.dtw.forward_with_cost_matrix(c_stu_stu.unsqueeze(0))
            t2t = self.dtw.forward_with_cost_matrix(c_tea_tea.unsqueeze(0))

            pair_loss = s2t - 0.5 * (s2s + t2t)
            total_loss = total_loss + pair_loss  # cộng trực tiếp, không cần view(1)

        if non_empty_pairs == 0:
            return torch.zeros(1, device=device, requires_grad=True)

        return total_loss
    '''

    def _get_target_embeddings(self, distiller, batch, pad_mask, teacher_pad_mask, model, teacher_model):
        target = batch["chosen_student_labels"]
        teacher_target = batch[f"chosen_teacher_labels"]
        
        if hasattr(model, "model") \
            and hasattr(model.model, "embed_tokens"):
            stu_embed_tokens = model.model.embed_tokens
        elif hasattr(model, "model") \
            and hasattr(model.model, "model") \
            and hasattr(model.model.model, "embed_tokens"):
            stu_embed_tokens = model.model.model.embed_tokens
        elif hasattr(model, "transformer") \
            and hasattr(model.transformer, "word_embeddings"):
            stu_embed_tokens = model.transformer.word_embeddings
        else:
            raise NotImplementedError

        if hasattr(teacher_model, "model") \
            and hasattr(teacher_model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.embed_tokens
        elif hasattr(teacher_model, "model") \
            and hasattr(teacher_model.model, "model") \
            and hasattr(teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.model.embed_tokens
        elif hasattr(teacher_model, "transformer") \
            and hasattr(teacher_model.model, "wte"):
            tea_embed_tokens = teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        stu_target_embeds = stu_embed_tokens(formal_target)

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        return stu_target_embeds, tea_target_embeds
