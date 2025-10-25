import torch
import torch.nn as nn
import torch.nn.functional as F
import C2f
import Conv

# models/yolo_scratch.py
# 從零實作 YOLOv8-nano（最小模型，適合從零訓練）

class CSPDarknet(nn.Module):
    """YOLOv8 Backbone - CSPDarknet53"""
    def __init__(self, width_mult=0.25, depth_mult=0.33):  # nano config
        super().__init__()
        # P1/2: [3, 640, 640] -> [64, 320, 320]
        self.stem = Conv(3, int(64*width_mult), k=3, s=2)
        
        # P2/4: [64, 320, 320] -> [128, 160, 160]
        self.stage1 = C2f(int(64*width_mult), int(128*width_mult), n=int(3*depth_mult))
        
        # P3/8: [128, 160, 160] -> [256, 80, 80]
        self.stage2 = C2f(int(128*width_mult), int(256*width_mult), n=int(6*depth_mult))
        
        # P4/16: [256, 80, 80] -> [512, 40, 40]
        self.stage3 = C2f(int(256*width_mult), int(512*width_mult), n=int(6*depth_mult))
        
        # P5/32: [512, 40, 40] -> [1024, 20, 20]
        self.stage4 = C2f(int(512*width_mult), int(1024*width_mult), n=int(3*depth_mult))

class PAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network"""
    # 實作 Top-down + Bottom-up 特徵融合

class DetectionHead(nn.Module):
    """YOLOv8 Detection Head with Long-Tail modifications"""
    def __init__(self, num_classes=4, use_decoupled=True):
        # Decoupled head: 分類和定位分支分離
        # 有利於 long-tail learning