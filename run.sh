# For mean-squared loss
for lr_denom in 100 200 300 400
do

python main.py \
    --dataset cifar10-5k \
    --batch_size 5000 \
    --lr_denom "$lr_denom" \
    --optim "RK4" \
    --model FCN \
    --epoch 30000 \
    --loss MSE

done

# For cross-entropy loss
for lr_denom in 10 20 50 80 110 200
do

python main.py \
    --dataset cifar10-5k \
    --batch_size 5000 \
    --lr_denom "$lr_denom" \
    --optim "RK4" \
    --model FCN \
    --loss CE

done
