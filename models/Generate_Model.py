from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Text import class_descriptor_5_only_face
from models.Adapter import Adapter
from clip import clip
from utils.utils import slerp
import copy
import itertools

class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        
        self.is_ensemble = any(isinstance(i, list) for i in input_text)
        
        if self.is_ensemble:
            self.num_classes = len(input_text)
            self.num_prompts_per_class = len(input_text[0])
            self.input_text = list(itertools.chain.from_iterable(input_text))
            print(f"=> Using Prompt Ensembling with {self.num_prompts_per_class} prompts per class.")
        else:
            self.input_text = input_text

        self.prompt_learner = PromptLearner(self.input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual

        # For EAA
        self.face_adapter = Adapter(c_in=512, reduction=4)

        # For MI Loss
        hand_crafted_prompts = class_descriptor_5_only_face
        self.tokenized_hand_crafted_prompts = torch.cat([clip.tokenize(p) for p in hand_crafted_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_hand_crafted_prompts).type(self.dtype)
        self.register_buffer("hand_crafted_prompt_embeddings", embedding)


        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        self.temporal_net_body = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        self.clip_model_ = clip_model
        self.project_fc = nn.Linear(1024, 512)

        # New classifier heads for Stage 2
        self.cls_bin = nn.Linear(512, 2)  # Binary head: Confusion vs. Non-Confusion
        self.cls_4 = nn.Linear(512, 4)   # 4-class head: for Non-Confusion classes
        
    def forward(self, image_face,image_body):
        ################# Visual Part #################
        # Face Part
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face.type(self.dtype))
        image_face_features = self.face_adapter(image_face_features) # Apply EAA
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(image_face_features)  # (4*512)
        
        # Body Part
        n, t, c, h, w = image_body.shape
        image_body = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder(image_body.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body(image_body_features)

        # Concatenate the two parts
        video_features = torch.cat((video_face_features, video_body_features), dim=-1)
        video_features = self.project_fc(video_features)
        
        # NOTE: The norm is applied after the heads for Stage 2
        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        ################# Stage 2 Heads #################
        # Calculate logits for the new heads BEFORE normalization
        logits_bin = self.cls_bin(video_features)
        logits_4 = self.cls_4(video_features)

        ################# Text Part & Original Head ###################
        # Normalize video features for CLIP-style classification
        video_features_norm = video_features / video_features.norm(dim=-1, keepdim=True)
        
        # Learnable prompts
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Hand-crafted prompts (for MI Loss, not used for classification)
        hand_crafted_prompts = self.hand_crafted_prompt_embeddings
        tokenized_hand_crafted_prompts = self.tokenized_hand_crafted_prompts.to(hand_crafted_prompts.device)
        hand_crafted_text_features = self.text_encoder(hand_crafted_prompts, tokenized_hand_crafted_prompts)
        hand_crafted_text_features = hand_crafted_text_features / hand_crafted_text_features.norm(dim=-1, keepdim=True)

        ################# Classification (Original 5-class) ###################
        # Calculate logits
        if self.is_ensemble:
            # Reshape text features for ensembling: (C*P, D) -> (C, P, D)
            text_features_reshaped = text_features.view(self.num_classes, self.num_prompts_per_class, -1)
            # Normalize again just in case (optional but safe)
            text_features_reshaped = text_features_reshaped / text_features_reshaped.norm(dim=-1, keepdim=True)
            
            # Compute logits per prompt: (B, D) @ (D, P, C) -> (B, P, C)
            # Note: We use einsum for clarity with batch and ensemble dimensions
            logits = torch.einsum('bd,cpd->bcp', video_features_norm, text_features_reshaped)
            
            # Average the logits across the prompts for each class
            output = torch.mean(logits, dim=2) / self.args.temperature

        elif self.args.slerp_weight > 0:
            video_features_expanded = video_features_norm.unsqueeze(1).expand(-1, hand_crafted_text_features.shape[0], -1)
            text_features_expanded = hand_crafted_text_features.unsqueeze(0).expand(video_features_norm.shape[0], -1, -1)
            
            instance_enhanced_text_features = slerp(text_features_expanded, video_features_expanded, self.args.slerp_weight)
            
            # Take the dot product between the video features and the instance-enhanced text features
            # We need to do this element-wise for each instance
            output = torch.einsum('bd,bcd->bc', video_features_norm, instance_enhanced_text_features) / self.args.temperature
        else:
            output = video_features_norm @ text_features.t() / self.args.temperature

        return output, logits_bin, logits_4, text_features, hand_crafted_text_features