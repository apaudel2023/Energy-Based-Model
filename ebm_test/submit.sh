LR=0.001
STEPSIZE=0.05
N_STEPS=100
N_EPOCH=500
ALPHA=1
MODEL=FCNet
DATA=2spirals

python3 train.py $DATA $MODEL --lr $LR --stepsize $STEPSIZE --n_steps $N_STEPS --n_epoch $N_EPOCH --alpha $ALPHA
