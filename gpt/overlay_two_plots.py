#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#两起点的叠加图
from PIL import Image
import os

def overlay_images(img_path1: str, img_path2: str, out_path: str, alpha: float = 0.5):
    """
    将两张同尺寸图片叠加，并保存到 out_path。
    alpha 控制第二张图的透明度（取值 0.0-1.0）。
    """
    # 打开并转为 RGBA，以便处理透明度
    img1 = Image.open(img_path1).convert("RGBA")
    img2 = Image.open(img_path2).convert("RGBA")

    # 如果两张图片大小不一致，就把第二张重采样到与第一张同样大小
    if img2.size != img1.size:
        img2 = img2.resize(img1.size, resample=Image.BILINEAR)

    # 把 img2 的 alpha 通道按 alpha 比例缩放
    r, g, b, a = img2.split()
    a = a.point(lambda i: int(i * alpha))
    img2 = Image.merge("RGBA", (r, g, b, a))

    # 新建一个透明画布，再把 img1、img2 依次粘贴进去
    composite = Image.new("RGBA", img1.size)
    composite.paste(img1, (0, 0))
    composite.paste(img2, (0, 0), mask=img2)

    # 转回 RGB 格式保存
    composite.convert("RGB").save(out_path, quality=95)
    print(f"Saved overlay image to: {out_path}")

if __name__ == "__main__":
    # —— 叠加 velocity 热图 —— 
    img1 = "output/vel_run1.jpg"
    img2 = "output/vel_run2.jpg"
    out = "output/vel_overlay.jpg"
    overlay_images(img1, img2, out, alpha=0.5)

    # —— 叠加 tau 热图 —— 
    img1 = "output/tau_run1.jpg"
    img2 = "output/tau_run2.jpg"
    out = "output/tau_overlay.jpg"
    overlay_images(img1, img2, out, alpha=0.5)
