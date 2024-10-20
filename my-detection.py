cd ~/jetson-inference/python/training/detection/ssd

python3 train_ssd.py --data=data/fruit --model-dir=models/fruit --batch-size=1 --workers=1 --epochs=1 --resume=models/trainedModel/mb1-ssd-Epoch-29-Loss-3.940372071041058.pth

python3 onnx_export.py --input=models/fruit/mb1-ssd-Epoch-29-Loss-3.940372071041058.pth --model-dir=models/fruit/

cd ~/jetson-inference/build/aarch64/bin

IMAGES=/home/nvidia/jetson-inference/data/images

detectnet --model=/home/nvidia/jetson-inference/python/training/detection/ssd/models/fruit/ssd-mobilenet.onnx --labels=/home/nvidia/jetson-inference/python/training/detection/ssd/models/fruit/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes "$IMAGES/fruit_*.jpg" $IMAGES/test/fruit_%i.jpg

cd ~/jetson-inference/build/aarch64/bin

detectnet --model=/home/nvidia/jetson-inference/python/training/detection/ssd/models/fruit/ssd-mobilenet.onnx --labels=/home/nvidia/jetson-inference/python/training/detection/ssd/models/fruit/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes/dev/video0

