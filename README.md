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

## Optimizations applied

- MobileNet backbone instead of VGG16
- Input resizing to 480px
- Region proposals reduced to 100
- Frame skipping (every 2nd frame)
- torch.no_grad() during inference

## Results (CPU)

VGG16 vs MobileNet:

- MobileNet: 59.3 FPS (smooth)
- VGG16: 0.5 FPS (unusable)

Score threshold:

- 0.3 → 4.1 FPS (more detections, more false positives)
- 0.5 → 4.2 FPS
- 0.6 → 59.3 FPS (best balance)
- 0.7 → 67.1 FPS (fewer detections)

IoU (NMS):

- 0.3 → 26.2 FPS (fewer overlapping boxes)
- 0.4 → 59.3 FPS (best balance)
- 0.6 → 247.2 FPS (more overlapping boxes kept)

Proposals:

- 50 → 47.5 FPS
- 100 → 59.3 FPS (best balance)
- 300 → 3.7 FPS

Best config: score=0.6, iou=0.4, proposals=100
