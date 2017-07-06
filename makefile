DATA_DIR=/local/scratch/ssd
TRAIN_DIR=/local/scratch/yaz21/tmp_vgg
run: vgg_multi_gpu_train.py
	CUDA_VISIBLE_DEVICES=2,3 python vgg_multi_gpu_train.py --num_preprocess_threads=4 --num_gpus=2 --train_dir=$(TRAIN_DIR) --data_dir=$(DATA_DIR)

git-add:
	git add -A
	git commit -m"vggnet first commit"
	git push

git-fetch:
	git fetch
	git merge
