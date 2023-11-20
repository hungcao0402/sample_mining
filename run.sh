### resnet50 ###
# market1501
strategy_pos=('moderate' 'hard' 'least' 'random')
strategy_neg=('hard' 'least' 'moderate' 'random')

for pos in "${strategy_pos[@]}"; do
    for neg in "${strategy_neg[@]}"; do
        CUDA_VISIBLE_DEVICES=1 python examples/train.py -b 64 -a resnet50 -d market1501 --iters 200 --eps 0.45 --num-instances 16 --pooling-type gem --memorybank CMhybrid_v2 --epochs 60  --negative_sample "$neg"  --positive_sample "$pos" --logs-dir "examples/logs/market1501/{$pos}_{$neg}"
    done
done

# for pos in "${strategy_pos[@]}"; do
#     for neg in "${strategy_neg[@]}"; do
#         CUDA_VISIBLE_DEVICES=4,5,6,7 python examples/train.py -b 256 -a resnet50 -d msmt17 --iters 300 --eps 0.7 --num-instances 16 --pooling-type gem --memorybank CMhybrid_v2 --epochs 60    \
#             --negative_sample "$pos"  --positive_sample "$neg" --logs-dir examples/logs/msmt17/{$pos}_{$neg}
#     done
# done

