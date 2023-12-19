import argparse
import os
import random

# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from functools import partial
from typing import Optional

os.system('python -m pip install Openmim')
os.system('python -m mim install mmengine')
os.system('python -m mim install "mmcv-full==1.6.2"')
os.system('python -m mim install "mmpose==0.29.0"')
os.system('python -m mim install "gradio==3.44.0"')
os.system('python setup.py develop')

import gradio as gr
import numpy as np
import torch
from PIL import ImageDraw
from matplotlib import pyplot as plt
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint
from mmpose.core import wrap_fp16_model
from mmpose.models import build_posenet
from torchvision import transforms

from demo import Resize_Pad
from models import *
from tools.visualization import str_is_int
import matplotlib

matplotlib.use('agg')


def plot_results(support_img, query_img, support_kp, support_w, query_kp,
                 query_w, skeleton,
                 initial_proposals, prediction, radius=6):
    h, w, c = support_img.shape
    prediction = prediction[-1].cpu().numpy() * h
    query_img = (query_img - np.min(query_img)) / (
            np.max(query_img) - np.min(query_img))
    for id, (img, w, keypoint) in enumerate(zip([query_img],
                                                [query_w],
                                                [prediction])):
        f, axes = plt.subplots()
        plt.imshow(img)
        for k in range(keypoint.shape[0]):
            if w[k] > 0:
                kp = keypoint[k, :2]
                c = (1, 0, 0, 0.75) if w[k] == 1 else (0, 0, 1, 0.6)
                patch = plt.Circle(kp, radius, color=c)
                axes.add_patch(patch)
                axes.text(kp[0], kp[1], k)
                plt.draw()
        for l, limb in enumerate(skeleton):
            kp = keypoint[:, :2]
            if l > len(COLORS) - 1:
                c = [x / 255 for x in random.sample(range(0, 255), 3)]
            else:
                c = [x / 255 for x in COLORS[l]]
            if w[limb[0]] > 0 and w[limb[1]] > 0:
                patch = plt.Line2D([kp[limb[0], 0], kp[limb[1], 0]],
                                   [kp[limb[0], 1], kp[limb[1], 1]],
                                   linewidth=6, color=c, alpha=0.6)
                axes.add_artist(patch)
        plt.axis('off')  # command for hiding the axis.
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        return plt


COLORS = [
    [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]
]

kp_src = []
skeleton = []
count = 0
color_idx = 0
prev_pt = None
prev_pt_idx = None
prev_clicked = None
original_support_image = None
checkpoint_path = ''

def process(query_img,
            cfg_path='configs/demo_b.py'):
    global skeleton
    cfg = Config.fromfile(cfg_path)
    kp_src_np = np.array(kp_src).copy().astype(np.float32)
    kp_src_np[:, 0] = kp_src_np[:, 0] / original_support_image.shape[
        0] * cfg.model.encoder_config.img_size
    kp_src_np[:, 1] = kp_src_np[:, 1] / original_support_image.shape[
        1] * cfg.model.encoder_config.img_size
    kp_src_np = np.flip(kp_src_np, 1).copy()
    kp_src_tensor = torch.tensor(kp_src_np).float()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(cfg.model.encoder_config.img_size,
                   cfg.model.encoder_config.img_size)])

    if len(skeleton) == 0:
        skeleton = [(0, 0)]

    support_img = preprocess(original_support_image).flip(0)[None]
    np_query = np.array(query_img)[:, :, ::-1].copy()
    q_img = preprocess(np_query).flip(0)[None]
    # Create heatmap from keypoints
    genHeatMap = TopDownGenerateTargetFewShot()
    data_cfg = cfg.data_cfg
    data_cfg['image_size'] = np.array([cfg.model.encoder_config.img_size,
                                       cfg.model.encoder_config.img_size])
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False
    kp_src_3d = torch.concatenate(
        (kp_src_tensor, torch.zeros(kp_src_tensor.shape[0], 1)), dim=-1)
    kp_src_3d_weight = torch.concatenate(
        (torch.ones_like(kp_src_tensor),
         torch.zeros(kp_src_tensor.shape[0], 1)), dim=-1)
    target_s, target_weight_s = genHeatMap._msra_generate_target(data_cfg,
                                                                 kp_src_3d,
                                                                 kp_src_3d_weight,
                                                                 sigma=1)
    target_s = torch.tensor(target_s).float()[None]
    target_weight_s = torch.ones_like(
        torch.tensor(target_weight_s).float()[None])

    data = {
        'img_s': [support_img],
        'img_q': q_img,
        'target_s': [target_s],
        'target_weight_s': [target_weight_s],
        'target_q': None,
        'target_weight_q': None,
        'return_loss': False,
        'img_metas': [{'sample_skeleton': [skeleton],
                       'query_skeleton': skeleton,
                       'sample_joints_3d': [kp_src_3d],
                       'query_joints_3d': kp_src_3d,
                       'sample_center': [kp_src_tensor.mean(dim=0)],
                       'query_center': kp_src_tensor.mean(dim=0),
                       'sample_scale': [
                           kp_src_tensor.max(dim=0)[0] -
                           kp_src_tensor.min(dim=0)[0]],
                       'query_scale': kp_src_tensor.max(dim=0)[0] -
                                      kp_src_tensor.min(dim=0)[0],
                       'sample_rotation': [0],
                       'query_rotation': 0,
                       'sample_bbox_score': [1],
                       'query_bbox_score': 1,
                       'query_image_file': '',
                       'sample_image_file': [''],
                       }]
    }
    # Load model
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.eval()
    with torch.no_grad():
        outputs = model(**data)
    # visualize results
    vis_s_weight = target_weight_s[0]
    vis_q_weight = target_weight_s[0]
    vis_s_image = support_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    vis_q_image = q_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    support_kp = kp_src_3d
    out = plot_results(vis_s_image,
                       vis_q_image,
                       support_kp,
                       vis_s_weight,
                       None,
                       vis_q_weight,
                       skeleton,
                       None,
                       torch.tensor(outputs['points']).squeeze(0),
                       )
    return out


with gr.Blocks() as demo:
    gr.Markdown('''
    # Pose Anything Demo
    We present a novel approach to category agnostic pose estimation that leverages the inherent geometrical relations between keypoints through a newly designed Graph Transformer Decoder. By capturing and incorporating this crucial structural information, our method enhances the accuracy of keypoint localization, marking a significant departure from conventional CAPE techniques that treat keypoints as isolated entities.
    ### [Paper](https://arxiv.org/abs/2311.17891) | [Official Repo](https://github.com/orhir/PoseAnything) 
    ## Instructions
    1. Upload an image of the object you want to pose on the **left** image.
    2. Click on the **left** image to mark keypoints.
    3. Click on the keypoints on the **right** image to mark limbs.
    4. Upload an image of the object you want to pose to the query image (**bottom**).
    5. Click **Evaluate** to pose the query image.
    ''')
    with gr.Row():
        support_img = gr.Image(label="Support Image",
                               type="pil",
                               info='Click to mark keypoints').style(
            height=400, width=400)
        posed_support = gr.Image(label="Posed Support Image",
                                 type="pil",
                                 interactive=False).style(height=400, width=400)
    with gr.Row():
        query_img = gr.Image(label="Query Image",
                             type="pil").style(height=400, width=400)
    with gr.Row():
        eval_btn = gr.Button(value="Evaluate")
    with gr.Row():
        output_img = gr.Plot(label="Output Image", height=400, width=400)


    def get_select_coords(kp_support,
                          limb_support,
                          evt: gr.SelectData,
                          r=0.015):
        global original_support_image
        if len(kp_src) == 0:
            original_support_image = np.array(kp_support.copy())[:, :,
                                     ::-1].copy()
        pixels_in_queue = set()
        pixels_in_queue.add((evt.index[1], evt.index[0]))
        while len(pixels_in_queue) > 0:
            pixel = pixels_in_queue.pop()
            if pixel[0] is not None and pixel[
                1] is not None and pixel not in kp_src:
                kp_src.append(pixel)
            else:
                print("Invalid pixel")
            if limb_support is None:
                canvas_limb = kp_support.copy()
            else:
                canvas_limb = limb_support.copy()
            canvas_kp = kp_support.copy()
            w, h = canvas_kp.size
            draw_pose = ImageDraw.Draw(canvas_kp)
            draw_limb = ImageDraw.Draw(canvas_limb)
            r = int(r * w)
            leftUpPoint = (pixel[1] - r, pixel[0] - r)
            rightDownPoint = (pixel[1] + r, pixel[0] + r)
            twoPointList = [leftUpPoint, rightDownPoint]
            draw_pose.ellipse(twoPointList, fill=(255, 0, 0, 255))
            draw_limb.ellipse(twoPointList, fill=(255, 0, 0, 255))

        return canvas_kp, canvas_limb


    def get_limbs(kp_support,
                  evt: gr.SelectData,
                  r=0.02, width=0.02):
        global count, color_idx, prev_pt, skeleton, prev_pt_idx, prev_clicked
        curr_pixel = (evt.index[1], evt.index[0])
        pixels_in_queue = set()
        pixels_in_queue.add((evt.index[1], evt.index[0]))
        canvas_kp = kp_support.copy()
        w, h = canvas_kp.size
        r = int(r * w)
        width = int(width * w)
        while len(pixels_in_queue) > 0 and curr_pixel != prev_clicked:
            pixel = pixels_in_queue.pop()
            prev_clicked = pixel
            closest_point = min(kp_src,
                                key=lambda p: (p[0] - pixel[0]) ** 2 +
                                              (p[1] - pixel[1]) ** 2)
            closest_point_index = kp_src.index(closest_point)
            draw_limb = ImageDraw.Draw(canvas_kp)
            if color_idx < len(COLORS):
                c = COLORS[color_idx]
            else:
                c = random.choices(range(256), k=3)
            leftUpPoint = (closest_point[1] - r, closest_point[0] - r)
            rightDownPoint = (closest_point[1] + r, closest_point[0] + r)
            twoPointList = [leftUpPoint, rightDownPoint]
            draw_limb.ellipse(twoPointList, fill=tuple(c))
            if count == 0:
                prev_pt = closest_point[1], closest_point[0]
                prev_pt_idx = closest_point_index
                count = count + 1
            else:
                if prev_pt_idx != closest_point_index:
                    # Create Line and add Limb
                    draw_limb.line([prev_pt, (closest_point[1], closest_point[0])],
                                   fill=tuple(c),
                                   width=width)
                    skeleton.append((prev_pt_idx, closest_point_index))
                    color_idx = color_idx + 1
                else:
                    draw_limb.ellipse(twoPointList, fill=(255, 0, 0, 255))
                count = 0
        return canvas_kp


    def set_qery(support_img):
        skeleton.clear()
        kp_src.clear()
        return support_img


    support_img.select(get_select_coords,
                       [support_img, posed_support],
                       [support_img, posed_support])
    support_img.upload(set_qery,
                       inputs=support_img,
                       outputs=posed_support)
    posed_support.select(get_limbs,
                         posed_support,
                         posed_support)
    eval_btn.click(fn=process,
                   inputs=[query_img],
                   outputs=output_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Anything Demo')
    parser.add_argument('--checkpoint',
                        help='checkpoint path',
                        default='checkpoints/demo.pth')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    demo.launch()
