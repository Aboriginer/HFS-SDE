#!/bin/bash
# sh train_fastMRI.sh "vp"

export CUDA_VISIBLE_DEVICES=3

if [ "$1" = "vp" ]
then
    echo "==================================================================="
    echo "================ run configs/vp/ddpm_continuous.py ================"
    echo "==================================================================="
    python main.py \
        --config=configs/vp/ddpm_continuous.py \
        --mode='train'  \
        --workdir=results
elif [ "$1" = "vp_ncsnpp" ]
then
    echo "========================================================================"
    echo "================ run configs/vp/ncsnpp_continuous.py    ================"
    echo "========================================================================"
    python main.py \
        --config=configs/vp/ncsnpp_continuous.py \
        --mode='train'  \
        --workdir=results
elif [ "$1" = "ve" ]
then
    echo "====================================================================="
    echo "================ run configs/ve/ncsnpp_continuous.py ================"
    echo "====================================================================="
    python main.py \
        --config=configs/ve/ncsnpp_continuous.py \
        --mode='train'  \
        --workdir=results
elif [ "$1" = "ve_ddpm" ]
then
    echo "======================================================================"
    echo "================ run configs/ve/ddpm_continuous.py    ================"
    echo "======================================================================"
    python main.py \
        --config=configs/ve/ddpm_continuous.py \
        --mode='train'  \
        --workdir=results
elif [ "$1" = "hfssde" ]
then
    echo "======================================================================"
    echo "================ run configs/hfssde/ddpm_continuous.py================"
    echo "======================================================================"
    python main.py \
        --config=configs/hfssde/ddpm_continuous.py \
        --mode='train'  \
        --workdir=results
else
    echo "==========================================================================================="
    echo "================ You must input one argument: ve, ve_ddpm, vp , vp_ncsnpp or hfssde ================"
    echo "==========================================================================================="
fi