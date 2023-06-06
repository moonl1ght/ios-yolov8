# Object detection and segmentation

Online object dtection and segmentation using [YOLOv8 by ultralytics](https://github.com/ultralytics/ultralytics).

The output of the YOLOv8 model processed on GPU using Metal.

1. Filtering bounding box and mask proposals with high confidence.

```swift
kernel void filterBBoxes(constant BBoxFilterParams& params [[ buffer(iParams) ]],
                         constant float* prediction [[ buffer(iPrediction) ]],
                         device atomic_int* bboxCount [[ buffer(iBBoxCount) ]],
                         device BBox* bboxes [[ buffer(iBBoxes) ]],
                         device float* maskProposals [[ buffer(iMaskProposals) ]],
                         uint2 gid [[thread_position_in_grid]])
{
  int stride = params.stride;
  float x = prediction[gid.x];
  float y = prediction[gid.x + stride];
  float w = prediction[gid.x + stride * 2];
  float h = prediction[gid.x + stride * 3];
  BBox box {
    .x = int((x - 0.5 * w) * params.factor.x),
    .y = int((y - 0.5 * h) * params.factor.y),
    .w = int(w * params.factor.x),
    .h = int(h * params.factor.y)
  };
  float maxConfidence = 0;
  int classId;
  uint length = NUM_OF_COORDINATES + params.numberOfClasses;
  for (uint i = NUM_OF_COORDINATES; i < length; ++i) {
    float confidence = prediction[gid.x + stride * i];
    if (confidence > maxConfidence) {
      maxConfidence = confidence;
      classId = i - NUM_OF_COORDINATES;
    }
  }
  if (maxConfidence > params.confidenceThreshold) {
    box.confidence = maxConfidence;
    box.classId = classId;
    int i = atomic_fetch_add_explicit(bboxCount, 1, memory_order_relaxed);
    if (params.hasSegmentationMask) {
      for (uint j = length; j < length + params.segmentationMaskLength; ++j) {
        float prop = prediction[gid.x + stride * j];
        maskProposals[i * params.segmentationMaskLength + j - length] = prop;
      }
    }
    bboxes[i] = box;
  }
}
```
2. Non maximum supression made on GPU.

```swift
kernel void nonMaximumSuppression(constant NMSParams& params [[ buffer(iParams) ]],
                                  constant BBox* bboxes [[ buffer(iBBoxes) ]],
                                  constant int& bboxCount [[ buffer(iBBoxCount) ]],
                                  device atomic_int* keptBBoxMap [[ buffer(iKeptBBoxMap) ]],
                                  uint2 gid [[thread_position_in_grid]])
{
  // Skip all the entries below the main diagonal and out of matrix bounds.
  if (gid.x >= uint(bboxCount) || gid.y >= uint(bboxCount) || gid.y >= gid.x) {
    return;
  }
  BBox box1 = bboxes[gid.x];
  BBox box2 = bboxes[gid.y];
  if (box1.classId == box2.classId) {
    // Intersection over union.
    float iou = IOU(bboxes[gid.x], bboxes[gid.y]);
    if (iou > params.iouThreshold) {
      if (box1.confidence > box2.confidence) {
        atomic_fetch_add_explicit(&keptBBoxMap[gid.x], 1, memory_order_relaxed);
      } else {
        atomic_fetch_add_explicit(&keptBBoxMap[gid.y], 1, memory_order_relaxed);
      }
    } else {
      atomic_fetch_add_explicit(&keptBBoxMap[gid.x], 1, memory_order_relaxed);
      atomic_fetch_add_explicit(&keptBBoxMap[gid.y], 1, memory_order_relaxed);
    }
  } else {
    atomic_fetch_add_explicit(&keptBBoxMap[gid.x], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&keptBBoxMap[gid.y], 1, memory_order_relaxed);
  }
}
```
