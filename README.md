# Demo
https://user-images.githubusercontent.com/36232582/230697286-15eb2b86-0853-4d28-b6df-d5591913a639.mp4

<h1>You need to reference the original yolov7 GitHub repo for installation instructions.</h1>
https://github.com/WongKinYiu/yolov7

<br>
<br>
<h2>Below is a cheat sheet I used for training and running inference</h2>

<pre>
<b>Example of a train:</b>
python train.py --workers 4 --device 0 --batch-size 8 --epochs 100 --img 640 640 --data data/custom_data.yaml --hyp data/hyp.scratch.tiny.yaml --cfg cfg/training/yolov7-tiny-custom.yaml --name yolov7-custom-tiny --weights yolov7-tiny.pt

python train.py --workers 4 --device 0 --batch-size 4 --epochs 100 --img 640 640 --data data/custom_data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt

<b>Detect:</b>
python detect.py --weights runs/train/yolov7-custom4/weights/best.pt --conf 0.5 --img-size 640 --source agility.mkv --view-img --no-trace

python detect.py --weights runs/train/yolov7-custom2/weights/best.pt --conf 0.5 --img-size 640 --source agility_3.mkv --view-img --no-trace
</pre>

<p>To run the bot, you will need to be in Prifddinas and currently on the course.</p>
