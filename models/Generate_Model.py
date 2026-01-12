from torch import nn
import torch
import itertools
from clip import clip

from models.Temporal_Model import Temporal_Transformer_Cls
from models.Prompt_Learner import PromptLearner, TextEncoder
from models.Text import class_descriptor_5_only_face
from models.Adapter import Adapter
from utils.utils import slerp


class GenerateModel(nn.Module):
    """
    Stage-2 model with:
    - CLIP-style 5-class head (for compatibility + MI/DC)
    - Binary head: Confusion vs Non-Confusion
    - 4-class head: Neutral / Enjoyment / Fatigue / Distraction
    """

    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args

        # ===== Prompt ensemble handling =====
        self.is_ensemble = any(isinstance(i, list) for i in input_text)
        if self.is_ensemble:
            self.num_classes = len(input_text)
            self.num_prompts_per_class = len(input_text[0])
            self.input_text = list(itertools.chain.from_iterable(input_text))
            print(f"=> Using Prompt Ensembling with {self.num_prompts_per_class} prompts per class.")
        else:
            self.input_text = input_text

        # ===== Text encoder =====
        self.prompt_learner = PromptLearner(self.input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual

        # ===== Adapter (EAA) =====
        self.face_adapter = Adapter(c_in=512, reduction=4)

        # ===== Hand-crafted prompts for MI =====
        hand_prompts = class_descriptor_5_only_face
        self.tokenized_hand_prompts = torch.cat([clip.tokenize(p) for p in hand_prompts])
        with torch.no_grad():
            emb = clip_model.token_embedding(self.tokenized_hand_prompts).type(self.dtype)
        self.register_buffer("hand_prompt_embeddings", emb)

        # ===== Temporal modeling =====
        self.temporal_net = Temporal_Transformer_Cls(
            num_patches=16, input_dim=512,
            depth=args.temporal_layers, heads=8,
            mlp_dim=1024, dim_head=64
        )
        self.temporal_net_body = Temporal_Transformer_Cls(
            num_patches=16, input_dim=512,
            depth=args.temporal_layers, heads=8,
            mlp_dim=1024, dim_head=64
        )

        # ===== Projection =====
        self.project_fc = nn.Linear(1024, 512)

        # ===== Stage-2 heads =====
        self.cls_bin = nn.Linear(512, 2)   # Confusion vs Non-confusion
        self.cls_4 = nn.Linear(512, 4)     # Neutral / Enjoyment / Fatigue / Distraction

    def forward(self, image_face, image_body):
        # ========== FACE ==========
        n, t, c, h, w = image_face.shape
        face = image_face.view(-1, c, h, w)
        face_feat = self.image_encoder(face.type(self.dtype))
        face_feat = self.face_adapter(face_feat)
        face_feat = face_feat.view(n, t, -1)
        face_vid = self.temporal_net(face_feat)

        # ========== BODY ==========
        n, t, c, h, w = image_body.shape
        body = image_body.view(-1, c, h, w)
        body_feat = self.image_encoder(body.type(self.dtype))
        body_feat = body_feat.view(n, t, -1)
        body_vid = self.temporal_net_body(body_feat)

        # ========== FUSION ==========
        video_feat = torch.cat([face_vid, body_vid], dim=-1)
        video_feat = self.project_fc(video_feat)

        # ========== STAGE-2 HEADS ==========
        logits_bin = self.cls_bin(video_feat)
        logits_4 = self.cls_4(video_feat)

        # ========== CLIP HEAD (5-class) ==========
        video_norm = video_feat / video_feat.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        text_feat = self.text_encoder(prompts, self.tokenized_prompts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        if self.is_ensemble:
            text_feat = text_feat.view(self.num_classes, self.num_prompts_per_class, -1)
            logits = torch.einsum("bd,cpd->bcp", video_norm, text_feat)
            output = logits.mean(dim=2) / self.args.temperature
        else:
            output = video_norm @ text_feat.t() / self.args.temperature

        # ===== Hand-crafted text (MI only) =====
        hc_tokens = self.tokenized_hand_prompts.to(video_feat.device)
        hc_feat = self.text_encoder(self.hand_prompt_embeddings, hc_tokens)
        hc_feat = hc_feat / hc_feat.norm(dim=-1, keepdim=True)

        return output, logits_bin, logits_4, text_feat, hc_feat