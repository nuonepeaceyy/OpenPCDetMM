import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel

import os
os.environ["TOKENIZERS_PARALLELISM"]="false"


class ConvFuser(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        
    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']
        cat_bev = torch.cat([img_bev,lidar_bev],dim=1)
        mm_bev = self.conv(cat_bev)
        batch_dict['spatial_features'] = mm_bev
        return batch_dict
    

class ConvFuserV2(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        self.conv_t = nn.Sequential(
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=1,  
                kernel_size=180,  
                stride=1,
                padding=0,
                bias=False
            ),  # [1,512,1,1] -> [1,1,180,180]
            nn.Tanh()
            )
        self.clip_model = CLIPModel.from_pretrained("/mnt/32THHD/zsj/project/mllms/clip")
        self.clip_processor = CLIPProcessor.from_pretrained("/mnt/32THHD/zsj/project/mllms/clip")
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']
        
        with torch.no_grad():  # 确保CLIP模型不参与梯度计算
            cat_input_ids = batch_dict['cat_input_ids'].tolist()
            cat_input_ids = self.clip_processor(cat_input_ids, return_tensors="pt", padding=True).input_ids
            cat_input_ids = cat_input_ids.cuda()
            text_feat = self.clip_model.get_text_features(cat_input_ids)
            
        text_feat = self.conv_t(text_feat)
        cat_bev = torch.cat([img_bev, lidar_bev, text_feat],dim=1)
        mm_bev = self.conv(cat_bev)
        batch_dict['spatial_features'] = mm_bev
        return batch_dict