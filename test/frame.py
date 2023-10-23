import imageio

# 其实就是1秒40个图片，每个图片旋转9°
t = imageio.mimread('blender_paper_lego_spiral_200000_rgb.mp4')

print(len(t))
