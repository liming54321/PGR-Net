CONFIG=configs/cfg_DGUNet_shadowFormer.py
OUTPUT_NAME=test_name
CKPT_NAME=""Pre training weight path""
INPUT_NAME=Datasets/ISTD/test/

MAX_FRAME=700

GPUS=1

OUTPUT_DIR=results/ISTD/${OUTPUT_NAME}

if [ ! -d ${OUTPUT_DIR} ];then
    mkdir -p ${OUTPUT_DIR}
    cp ${CONFIG} ${OUTPUT_DIR}
    cp ${0} ${OUTPUT_DIR}
fi

infer() {
    # $1 output
    # $2 bmp gain

    if [ ! -d ${OUTPUT_DIR}/${1} ];then
        mkdir -p ${OUTPUT_DIR}/${1}
    fi
    # copy isp info
    # cp ${1}/*.txt ${OUTPUT_DIR}/${2}

    # infer
    python infer_ISTD_image.py \
        -c ${CONFIG} \
        -o ${OUTPUT_DIR}/${1} \
        --max-frame-num ${MAX_FRAME} \
        --gpus ${GPUS} \
        --ckpt ${CKPT_NAME} \
        -i ${INPUT_NAME}


}

infer ${OUTPUT_NAME}

