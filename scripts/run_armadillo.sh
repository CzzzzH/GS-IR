python train.py \
-m outputs/armadillo/ \
-s datasets/TensoIR/armadillo/ \
--iterations 30000 \
--eval

python baking.py \
-m outputs/armadillo/ \
--checkpoint outputs/armadillo/chkpnt30000.pth \
--bound 1.5 \
--occlu_res 128 \
--occlusion 0.25

python train.py \
-m outputs/armadillo/ \
-s datasets/TensoIR/armadillo/ \
--start_checkpoint outputs/armadillo/chkpnt30000.pth \
--iterations 35000 \
--gamma \
--indirect

python normal_eval.py \
--gt_dir datasets/TensoIR/armadillo/ \
--output_dir outputs/armadillo/test/ours_None > logs/armadillo_normal.txt

python render.py \
-m outputs/armadillo \
-s datasets/TensoIR/armadillo/ \
--checkpoint outputs/armadillo/chkpnt35000.pth \
--eval \
--skip_train \
--pbr \
--gamma \
--indirect > logs/armadillo_novel_view_synthesis.txt

python render.py \
-m outputs/armadillo \
-s datasets/TensoIR/armadillo/ \
--checkpoint outputs/armadillo/chkpnt35000.pth \
--eval \
--skip_train \
--gamma \
--brdf_eval > logs/armadillo_albedo.txt

python relight.py \
-m outputs/armadillo \
-s datasets/TensoIR/armadillo/ \
--checkpoint outputs/armadillo/chkpnt35000.pth \
--hdri datasets/TensoIR/Environment_Maps/high_res_envmaps_2k/bridge.hdr \
--eval \
--gamma

python relight.py \
-m outputs/armadillo \
-s datasets/TensoIR/armadillo/ \
--checkpoint outputs/armadillo/chkpnt35000.pth \
--hdri datasets/TensoIR/Environment_Maps/high_res_envmaps_2k/city.hdr \
--eval \
--gamma

python relight.py \
-m outputs/armadillo \
-s datasets/TensoIR/armadillo/ \
--checkpoint outputs/armadillo/chkpnt35000.pth \
--hdri datasets/TensoIR/Environment_Maps/high_res_envmaps_2k/fireplace.hdr \
--eval \
--gamma

python relight.py \
-m outputs/armadillo \
-s datasets/TensoIR/armadillo/ \
--checkpoint outputs/armadillo/chkpnt35000.pth \
--hdri datasets/TensoIR/Environment_Maps/high_res_envmaps_2k/forest.hdr \
--eval \
--gamma

python relight.py \
-m outputs/armadillo \
-s datasets/TensoIR/armadillo/ \
--checkpoint outputs/armadillo/chkpnt35000.pth \
--hdri datasets/TensoIR/Environment_Maps/high_res_envmaps_2k/night.hdr \
--eval \
--gamma

python relight_eval.py \
--output_dir outputs/armadillo/test/ours_None/relight/ \
--gt_dir datasets/TensoIR/armadillo/ > logs/armadillo_relight.txt