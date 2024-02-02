import argparse
import random

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmpose.core import wrap_fp16_model
from mmpose.models import build_posenet
from torchvision import transforms

from demo import Resize_Pad
from models import *

# Copyright (c) OpenMMLab. All rights reserved.
# os.system('python -m pip install timm')
# os.system('python -m pip install Openmim')
# os.system('python -m mim install mmengine')
# os.system('python -m mim install "mmcv-full==1.6.2"')
# os.system('python -m mim install "mmpose==0.29.0"')
# os.system('python -m mim install "gradio==3.44.0"')
# os.system('python setup.py develop')

matplotlib.use('agg')
checkpoint_path = ''


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


def process(query_img, state,
            cfg_path='configs/demo_b.py'):
    cfg = Config.fromfile(cfg_path)
    width, height, _ = state['original_support_image'].shape
    kp_src_np = np.array(state['kp_src']).copy().astype(np.float32)
    kp_src_np[:, 0] = kp_src_np[:, 0] / (
            width // 4) * cfg.model.encoder_config.img_size
    kp_src_np[:, 1] = kp_src_np[:, 1] / (
            height // 4) * cfg.model.encoder_config.img_size
    kp_src_np = np.flip(kp_src_np, 1).copy()
    kp_src_tensor = torch.tensor(kp_src_np).float()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(cfg.model.encoder_config.img_size,
                   cfg.model.encoder_config.img_size)])

    if len(state['skeleton']) == 0:
        state['skeleton'] = [(0, 0)]

    support_img = preprocess(state['original_support_image']).flip(0)[None]
    np_query = np.array(query_img)[:, :, ::-1].copy()
    q_img = preprocess(np_query).flip(0)[None]
    # Create heatmap from keypoints
    genHeatMap = TopDownGenerateTargetFewShot()
    data_cfg = cfg.data_cfg
    data_cfg['image_size'] = np.array([cfg.model.encoder_config.img_size,
                                       cfg.model.encoder_config.img_size])
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False
    kp_src_3d = torch.cat(
        (kp_src_tensor, torch.zeros(kp_src_tensor.shape[0], 1)), dim=-1)
    kp_src_3d_weight = torch.cat(
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
        'img_metas': [{'sample_skeleton': [state['skeleton']],
                       'query_skeleton': state['skeleton'],
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
                       state['skeleton'],
                       None,
                       torch.tensor(outputs['points']).squeeze(0),
                       )
    return out, state


def update_examples(support_img, posed_support, query_img, state, r=0.015, width=0.02):
    state['color_idx'] = 0
    state['original_support_image'] = np.array(support_img)[:, :, ::-1].copy()
    support_img, posed_support, _ = set_query(support_img, state, example=True)
    w, h = support_img.size
    draw_pose = ImageDraw.Draw(support_img)
    draw_limb = ImageDraw.Draw(posed_support)
    r = int(r * w)
    width = int(width * w)
    for pixel in state['kp_src']:
        leftUpPoint = (pixel[1] - r, pixel[0] - r)
        rightDownPoint = (pixel[1] + r, pixel[0] + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw_pose.ellipse(twoPointList, fill=(255, 0, 0, 255))
        draw_limb.ellipse(twoPointList, fill=(255, 0, 0, 255))
    for limb in state['skeleton']:
        point_a = state['kp_src'][limb[0]][::-1]
        point_b = state['kp_src'][limb[1]][::-1]
        if state['color_idx'] < len(COLORS):
            c = COLORS[state['color_idx']]
            state['color_idx'] += 1
        else:
            c = random.choices(range(256), k=3)
        draw_limb.line([point_a, point_b], fill=tuple(c), width=width)
    return support_img, posed_support, query_img, state


def get_select_coords(kp_support,
                      limb_support,
                      state,
                      evt: gr.SelectData,
                      r=0.015):
    pixels_in_queue = set()
    pixels_in_queue.add((evt.index[1], evt.index[0]))
    while len(pixels_in_queue) > 0:
        pixel = pixels_in_queue.pop()
        if pixel[0] is not None and pixel[1] is not None and pixel not in \
                state['kp_src']:
            state['kp_src'].append(pixel)
        else:
            continue
        if limb_support is None:
            canvas_limb = kp_support
        else:
            canvas_limb = limb_support
        canvas_kp = kp_support
        w, h = canvas_kp.size
        draw_pose = ImageDraw.Draw(canvas_kp)
        draw_limb = ImageDraw.Draw(canvas_limb)
        r = int(r * w)
        leftUpPoint = (pixel[1] - r, pixel[0] - r)
        rightDownPoint = (pixel[1] + r, pixel[0] + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw_pose.ellipse(twoPointList, fill=(255, 0, 0, 255))
        draw_limb.ellipse(twoPointList, fill=(255, 0, 0, 255))
    return canvas_kp, canvas_limb, state


def get_limbs(kp_support,
              state,
              evt: gr.SelectData,
              r=0.02, width=0.02):
    curr_pixel = (evt.index[1], evt.index[0])
    pixels_in_queue = set()
    pixels_in_queue.add((evt.index[1], evt.index[0]))
    canvas_kp = kp_support
    w, h = canvas_kp.size
    r = int(r * w)
    width = int(width * w)
    while len(pixels_in_queue) > 0 and curr_pixel != state['prev_clicked']:
        pixel = pixels_in_queue.pop()
        state['prev_clicked'] = pixel
        closest_point = min(state['kp_src'],
                            key=lambda p: (p[0] - pixel[0]) ** 2 +
                                          (p[1] - pixel[1]) ** 2)
        closest_point_index = state['kp_src'].index(closest_point)
        draw_limb = ImageDraw.Draw(canvas_kp)
        if state['color_idx'] < len(COLORS):
            c = COLORS[state['color_idx']]
        else:
            c = random.choices(range(256), k=3)
        leftUpPoint = (closest_point[1] - r, closest_point[0] - r)
        rightDownPoint = (closest_point[1] + r, closest_point[0] + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw_limb.ellipse(twoPointList, fill=tuple(c))
        if state['count'] == 0:
            state['prev_pt'] = closest_point[1], closest_point[0]
            state['prev_pt_idx'] = closest_point_index
            state['count'] = state['count'] + 1
        else:
            if state['prev_pt_idx'] != closest_point_index:
                # Create Line and add Limb
                draw_limb.line(
                    [state['prev_pt'], (closest_point[1], closest_point[0])],
                    fill=tuple(c),
                    width=width)
                state['skeleton'].append(
                    (state['prev_pt_idx'], closest_point_index))
                state['color_idx'] = state['color_idx'] + 1
            else:
                draw_limb.ellipse(twoPointList, fill=(255, 0, 0, 255))
            state['count'] = 0
    return canvas_kp, state


def set_query(support_img, state, example=False):
    if not example:
        state['skeleton'].clear()
        state['kp_src'].clear()
    state['original_support_image'] = np.array(support_img)[:, :, ::-1].copy()
    width, height = support_img.size
    support_img = support_img.resize((width // 4, width // 4),
                                     Image.Resampling.LANCZOS)
    return support_img, support_img, state


with gr.Blocks() as demo:
    state = gr.State({
        'kp_src': [],
        'skeleton': [],
        'count': 0,
        'color_idx': 0,
        'prev_pt': None,
        'prev_pt_idx': None,
        'prev_clicked': None,
        'original_support_image': None,
    })

    gr.Markdown('''
    # Pose Anything Demo
    We present a novel approach to category agnostic pose estimation that 
    leverages the inherent geometrical relations between keypoints through a 
    newly designed Graph Transformer Decoder. By capturing and incorporating 
    this crucial structural information, our method enhances the accuracy of 
    keypoint localization, marking a significant departure from conventional 
    CAPE techniques that treat keypoints as isolated entities.
    ### [Paper](https://arxiv.org/abs/2311.17891) | [Official Repo](https://github.com/orhir/PoseAnything) 
    ## Instructions
    1. Upload an image of the object you want to pose on the **left** image.
    2. Click on the **left** image to mark keypoints.
    3. Click on the keypoints on the **right** image to mark limbs.
    4. Upload an image of the object you want to pose to the query image (
    **bottom**).
    5. Click **Evaluate** to pose the query image.
    ''')
    with gr.Row():
        support_img = gr.Image(label="Support Image",
                               type="pil",
                               info='Click to mark keypoints').style(
            height=400, width=400)
        posed_support = gr.Image(label="Posed Support Image",
                                 type="pil",
                                 interactive=False).style(height=400,
                                                          width=400)
    with gr.Row():
        query_img = gr.Image(label="Query Image",
                             type="pil").style(height=400, width=400)
    with gr.Row():
        eval_btn = gr.Button(value="Evaluate")
    with gr.Row():
        output_img = gr.Plot(label="Output Image", height=400, width=400)
    with gr.Row():
        gr.Markdown("## Examples")
    with gr.Row():
        gr.Examples(
            examples=[
                ['examples/dog2.png',
                 'examples/dog2.png',
                 'examples/dog1.png',
                 {'kp_src': [(50, 58), (51, 78), (66, 57), (118, 79),
                             (154, 79), (217, 74), (218, 103), (156, 104),
                             (152, 151), (215, 162), (213, 191),
                             (152, 174), (108, 171)],
                  'skeleton': [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5),
                               (3, 7), (7, 6), (3, 12), (12, 8), (8, 9),
                               (12, 11), (11, 10)], 'count': 0,
                  'color_idx': 0, 'prev_pt': (174, 152),
                  'prev_pt_idx': 11, 'prev_clicked': (207, 186),
                  'original_support_image': None,
                  }
                 ],
                ['examples/sofa1.jpg',
                 'examples/sofa1.jpg',
                 'examples/sofa2.jpg',
                 {
                     'kp_src': [(82, 28), (65, 30), (52, 26), (65, 50),
                                (84, 52), (53, 54), (43, 52), (45, 71),
                                (81, 69), (77, 39), (57, 43), (58, 64),
                                (46, 42), (49, 65)],
                     'skeleton': [(0, 1), (3, 1), (3, 4), (10, 9), (11, 8),
                                  (1, 10), (10, 11), (11, 3), (1, 2), (7, 6),
                                  (5, 13), (5, 3), (13, 11), (12, 10), (12, 2),
                                  (6, 10), (7, 11)], 'count': 0,
                     'color_idx': 23, 'prev_pt': (71, 45), 'prev_pt_idx': 7,
                     'prev_clicked': (56, 63),
                     'original_support_image': None,
                 }],
                ['examples/person1.jpeg',
                 'examples/person1.jpeg',
                 'examples/person2.jpeg',
                 {
                     'kp_src': [(121, 95), (122, 160), (154, 130), (184, 106),
                                (181, 153)],
                     'skeleton': [(0, 1), (1, 2), (0, 2), (2, 3), (2, 4),
                                  (4, 3)], 'count': 0, 'color_idx': 6,
                     'prev_pt': (153, 181), 'prev_pt_idx': 4,
                     'prev_clicked': (181, 108),
                     'original_support_image': None,
                 }]
            ],
            inputs=[support_img, posed_support, query_img, state],
            outputs=[support_img, posed_support, query_img, state],
            fn=update_examples,
            run_on_click=True,
        )

    support_img.select(get_select_coords,
                       [support_img, posed_support, state],
                       [support_img, posed_support, state])
    support_img.upload(set_query,
                       inputs=[support_img, state],
                       outputs=[support_img, posed_support, state])
    posed_support.select(get_limbs,
                         [posed_support, state],
                         [posed_support, state])
    eval_btn.click(fn=process,
                   inputs=[query_img, state],
                   outputs=[output_img, state])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Anything Demo')
    parser.add_argument('--checkpoint',
                        help='checkpoint path',
                        default='https://github.com/orhir/PoseAnything'
                                '/releases/download/1.0.0/demo_b.pth')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    demo.launch()
