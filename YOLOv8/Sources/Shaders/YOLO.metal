//
// Created by moonl1ght 27.02.2023.
//

#include <metal_stdlib>
#import "YOLO.h"

using namespace metal;

// Intersection over union.
float IOU(BBox box1, BBox box2) {
  float x1 = max(box1.x, box2.x);
  float y1 = max(box1.y, box2.y);
  float x2 = min(box1.x + box1.w, box2.x + box2.w);
  float y2 = min(box1.y + box1.h, box2.y + box2.h);
  float w = max(y2 - y1, 0.0);
  float h = max(x2 - x1, 0.0);
  float intersection = w * h;
  float a1 = box1.w * box1.h;
  float a2 = box2.w * box2.h;
  return intersection / (a1 + a2 - intersection);
}

// Non maximum suppression.
kernel void NMS(constant NMSParams& params [[ buffer(0) ]],
                constant BBox* bboxes [[ buffer(1) ]],
                constant int& bboxCount [[ buffer(2) ]],
                device atomic_int* keptBBoxMap [[ buffer(3) ]],
                uint2 gid [[thread_position_in_grid]])
{
  // Skip all the entries below the main diagonal and out of matrix bounds.
  if (gid.x >= uint(bboxCount) || gid.y >= uint(bboxCount) || gid.y >= gid.x) {
    return;
  }
  BBox box1 = bboxes[gid.x];
  BBox box2 = bboxes[gid.y];
  if (box1.classId == box2.classId) {
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

kernel void computeThreadgroupsPerGrid(constant int& bboxCount [[ buffer(0) ]],
                                       constant int* threadgroupSize [[ buffer(1) ]],
                                       device int* threadgroupsPerGrid [[ buffer(2) ]])
{
  threadgroupsPerGrid[0] = (bboxCount + threadgroupSize[0] - 1) / threadgroupSize[0];
  threadgroupsPerGrid[1] = (bboxCount + threadgroupSize[1] - 1) / threadgroupSize[1];
  threadgroupsPerGrid[2] = 1;
}

kernel void sigmoid(constant float* input [[ buffer(0) ]],
                    device float* output [[ buffer(1) ]],
                    uint2 gid [[thread_position_in_grid]],
                    uint2 grid_size [[ threads_per_grid ]])
{
  uint pos = gid.x * grid_size.y + gid.y;
  float x = 1 / (1 + exp(-input[pos]));
  if (x > 0.5) {
    output[pos] = 1;
  } else {
    output[pos] = 0;
  }
}

kernel void cleanUpBuffers(device BBox* bboxes [[ buffer(0) ]],
                           device int* keptBBoxMap [[ buffer(1) ]],
                           uint2 gid [[thread_position_in_grid]])
{
  bboxes[gid.x] = BBox();
  keptBBoxMap[gid.x] = 0;
}

// Filter bounding boxes with high confidence.
kernel void filterBBoxes(constant BBoxFilterParams& params [[ buffer(0) ]],
                         constant float* prediction [[ buffer(1) ]],
                         device atomic_int* bboxCount [[ buffer(2) ]],
                         device BBox* bboxes [[ buffer(3) ]],
                         device float* maskProposals [[ buffer(4) ]],
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
  uint length = 4 + params.numberOfClasses;
  for (uint i = 4; i < length; ++i) {
    float confidence = prediction[gid.x + stride * i];
    if (confidence > maxConfidence) {
      maxConfidence = confidence;
      classId = i - 4;
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
