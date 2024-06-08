python train.py \
-m outputs/bicycle/ \
-s datasets/MIP/bicycle/ \
--iterations 30000 \
-i images_4 \
-r 1 \
--lambda_normal 0.03 \
--densify_grad_threshold 0.0002

python baking.py \
-m outputs/bicycle/ \
--checkpoint outputs/bicycle/chkpnt30000.pth \
--bound 16.0 \
--occlu_res 256 \
--occlusion 0.4

python train.py \
-m outputs/bicycle \
-s datasets/MIP/bicycle/ \
--start_checkpoint outputs/bicycle/chkpnt30000.pth \
--iterations 40000 \
-i images_4 \
-r 1 \
--metallic \
--indirect

python render.py \
-m outputs/bicycle \
-s datasets/MIP/bicycle/ \
--checkpoint outputs/bicycle/chkpnt40000.pth \
-i images_4 \
-r 1 \
--eval \
--skip_train \
--pbr \
--metallic \
--indirect > logs/bicycle_novel_view_synthesis.txt

python relight.py \
-m outputs/bicycle \
-s datasets/MIP/bicycle/ \
--checkpoint outputs/bicycle/chkpnt40000.pth \
--hdri datasets/TensoIR/Environment_Maps/high_res_envmaps_2k/snow.hdr \
--eval \
--gamma
