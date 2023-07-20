# When Evaling with multiple GPUs, model needs return Tensors instead of list or dict.
GPU='0'
TESTSET='isomer_swin_tiny'
ENCODER='swin_tiny'
IMG_SIZE=512
NUM_POINTS=121
INFER_DATASET='DAVIS'
INFER_MODEL_PATH='./checkpoints/isomer.pth'
INFER_SAVE='./test_results/'$TESTSET'/'
VAL_BATCH_SIZE=16
INFER_DATASET_PATH='../dataset/TestSet/'

python tools/inference.py \
	--gpu $GPU \
	--encoder $ENCODER \
	--num_point $NUM_POINTS \
	--img_size $IMG_SIZE \
	--infer_dataset $INFER_DATASET \
	--infer_model_path $INFER_MODEL_PATH \
	--infer_save $INFER_SAVE \
	--val_batchsize $VAL_BATCH_SIZE \
	--infer_dataset_path $INFER_DATASET_PATH
