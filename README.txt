将用于训练的四张大图（382、182以及他们的标签）放入data文件夹中，然后运行run.sh。
注：run.sh文件包含切图、划分训练集、验证集（python cut_data.py）和训练模型(CUDA_VISIBLE_DEVICES=0 python train.py --backbone=hrnet --batchsize=4 --lr=0.01 --num_epochs=150)
