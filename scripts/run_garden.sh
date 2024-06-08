python train.py \
-m outputs/garden/ \
-s datasets/MIP/garden/ \
--iterations 30000 \
-i images_4 \
-r 1 \
--lambda_normal 0.03 \
--densify_grad_threshold 0.0002

python baking.py \
-m outputs/garden/ \
--checkpoint outputs/garden/chkpnt30000.pth \
--bound 16.0 \
--occlu_res 256 \
--occlusion 0.4

python train.py \
-m outputs/garden \
-s datasets/MIP/garden/ \
--start_checkpoint outputs/garden/chkpnt30000.pth \
--iterations 40000 \
-i images_4 \
-r 1 \
--metallic \
--indirect 

python render.py \
-m outputs/garden \
-s datasets/MIP/garden/ \
--checkpoint outputs/garden/chkpnt40000.pth \
-i images_4 \
-r 1 \
--eval \
--skip_train \
--pbr \
--metallic \
--indirect > logs/garden_novel_view_synthesis.txt

python relight.py \
-m outputs/garden \
-s datasets/MIP/garden/ \
--checkpoint outputs/garden/chkpnt40000.pth \
--hdri datasets/TensoIR/Environment_Maps/high_res_envmaps_2k/courtyard.hdr \
--eval \
--gamma
