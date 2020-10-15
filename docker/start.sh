sudo docker run -d -p 2222:22 --rm --gpus all \
	-e  NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
	-v ~/workspace:/root/workspace \
	--name ranix chenzhekl/ranix

