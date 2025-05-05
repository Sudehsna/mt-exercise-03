#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

models=$base/models
configs=$base/configs

mkdir -p $models

num_threads=8
#device=0

# measure time

SECONDS=0

logs=$base/logs

model_name=deen_transformer_postnorm # OR model_name=deen_transformer_prenorm

# Finding joeynmt directory
joeynmt_dir=$base/../joeynmt
export PYTHONPATH=$joeynmt_dir:$PYTHONPATH

mkdir -p $logs

mkdir -p $logs/$model_name

OMP_NUM_THREADS=$num_threads python -m joeynmt train $configs/$model_name.yaml > $logs/$model_name/out_post 2> $logs/$model_name/err

echo "time taken:"
echo "$SECONDS seconds"