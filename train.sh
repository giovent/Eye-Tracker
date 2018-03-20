rm -r Tensorboard/*
CUDA_VISIBLE_DEVICES=0 python model_trainer.py & tensorboard --logdir=Tensorboard --port=5005
