# rcnn-webcam

Real-time object detection using pretrained Faster R-CNN with webcam. Saves output to .avi file.

## Usage

```
python webcam_rcnn.py --model mobilenet
python webcam_rcnn.py --model vgg16
python webcam_rcnn.py --score_thresh 0.3
python webcam_rcnn.py --iou_thresh 0.3
python webcam_rcnn.py --proposals 50
```

## Optimizations

- MobileNet backbone instead of VGG16
- Input resizing to 480px
- Region proposals reduced to 100
- Frame skipping (every 2nd frame)
- torch.no_grad() during inference

## Results (CPU)

VGG16 vs MobileNet:

- MobileNet: 7.8 FPS
- VGG16: 0.9 FPS (~8.7x slower)

Score threshold:

- 0.3 -> 8.3 FPS, more detections, more false positives
- 0.5 -> 8.1 FPS
- 0.6 -> 7.8 FPS (best balance)
- 0.7 -> 8.0 FPS, fewer detections

IoU (NMS):

- 0.3 -> 7.8 FPS, fewer overlapping boxes
- 0.4 -> 7.8 FPS (best balance)
- 0.6 -> 8.0 FPS, more overlapping boxes

Proposals:

- 50 -> 8.8 FPS, slight accuracy drop
- 100 -> 7.8 FPS (best balance)
- 300 -> 6.1 FPS, slower

Best config: score=0.6, iou=0.4, proposals=100
