#!/bin/sh

python training.py | tee logs/training.log
python makefig_trj.py | tee logs/makefig_trj.log
python umap_train_single.py | tee logs/umap_train_single.log
python umap_train_many.py | tee logs/umap_train_many.log
python eval_pre_rec.py | tee logs/eval_pre_rec.log
python eval_many_pre_rec.py | tee logs/eval_many_pre_rec.log
python lyapunov.py | tee logs/lyapunov.log
python makefig_lyapunov.py | tee logs/makefig_lyapunov.log
python lyapunov_direct.py | tee logs/lyapunov_direct.log

