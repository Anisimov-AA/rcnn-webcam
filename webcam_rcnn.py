import cv2
import torch
import torchvision
from torchvision import transforms
import time
import argparse

# choose backbone from command line: python webcam_rcnn.py --model vgg16
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet", choices=["vgg16", "mobilenet"])
parser.add_argument("--score_thresh", type=float, default=0.6)
parser.add_argument("--iou_thresh", type=float, default=0.4)
parser.add_argument("--proposals", type=int, default=100)
parser.add_argument("--skip_frames", type=int, default=2)
parser.add_argument("--resize", type=int, default=480)
args = parser.parse_args()

# load pretrained faster rcnn
if args.model == "vgg16":
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    print("Loaded Faster R-CNN with ResNet50 (heavier, similar to VGG16)")
else:
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT",
        rpn_post_nms_top_n_test=args.proposals
    )
    print(f"Loaded Faster R-CNN with MobileNet (proposals={args.proposals})")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Running on: {device}")

# COCO class names (pretrained model detects 91 classes)
COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# colors for different classes
colors = {}

def get_color(label):
    if label not in colors:
        # generate a random but consistent color per class
        torch.manual_seed(hash(label) % 1000)
        colors[label] = tuple(int(x) for x in torch.randint(100, 255, (3,)))
    return colors[label]


def run_detection(frame):
    # resize for speed
    h, w = frame.shape[:2]
    scale = args.resize / max(h, w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # convert to tensor
    img = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.to(device)

    with torch.no_grad():
        results = model([img])[0]

    return results, scale


def draw_boxes(frame, results, scale):
    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    # apply NMS with custom iou threshold
    keep = torchvision.ops.nms(
        torch.tensor(boxes), torch.tensor(scores), args.iou_thresh
    )
    boxes = boxes[keep.numpy()]
    labels = labels[keep.numpy()]
    scores = scores[keep.numpy()]

    for box, label, score in zip(boxes, labels, scores):
        if score < args.score_thresh:
            continue

        # scale box back to original frame size
        x1, y1, x2, y2 = (box / scale).astype(int)
        name = COCO_NAMES[label] if label < len(COCO_NAMES) else "?"
        color = get_color(name)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{name}: {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


# open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# setup video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (w, h))

print(f"\nSettings: model={args.model}, score={args.score_thresh}, "
      f"iou={args.iou_thresh}, proposals={args.proposals}, "
      f"skip={args.skip_frames}, resize={args.resize}")
print("Press 'q' to quit\n")

frame_count = 0
fps_list = []
last_results = None
last_scale = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # skip frames for speed - reuse previous detections
    if frame_count % args.skip_frames == 0:
        last_results, last_scale = run_detection(frame)

    if last_results is not None:
        frame = draw_boxes(frame, last_results, last_scale)

    elapsed = time.time() - start
    fps = 1.0 / elapsed if elapsed > 0 else 0
    fps_list.append(fps)

    # show fps on frame
    cv2.putText(frame, f"FPS: {fps:.1f} [{args.model}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("R-CNN Webcam Detection", frame)
    out.write(frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
print(f"\nDone! {frame_count} frames processed")
print(f"Average FPS: {avg_fps:.1f}")
print(f"Video saved to output.avi")