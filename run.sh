python train.py \
-m outputs/lego/ \
-s lego_tensor \
--iterations 30000 \
--eval


python baking.py \
-m outputs/lego/ \
--checkpoint outputs/lego/chkpnt30000.pth \
--bound 1.5 \
--occlu_res 128 \
--occlusion 0.25

python train.py \
-m outputs/lego/ \
-s lego_tensor/ \
--start_checkpoint outputs/lego/chkpnt30000.pth \
--iterations 35000 \
--eval \
--gamma \
--indirect \
--tone \
--metallic
