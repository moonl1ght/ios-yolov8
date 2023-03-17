//
// Created by moonl1ght 27.02.2023.
//

#include <metal_stdlib>
#import "Common.h"
#import "YOLO.h"

using namespace metal;

typedef struct {
  float2 position [[ attribute(kVertexAttributePosition) ]];
  float2 uv [[ attribute(kVertexAttributeUV) ]];
} VertexIn;

typedef struct {
  float4 position [[ position ]];
  float2 uv;
} FragmentIn;

typedef FragmentIn VertexOut;

vertex VertexOut vertexBaseRendering(const VertexIn in [[ stage_in ]],
                                     constant VertexUniforms& uniforms [[ buffer(1) ]])
{
  VertexOut out {
    .position = float4(in.position, 0.0, 1.0) * uniforms.scaleMatrix,
    .uv = in.uv
  };
  return out;
}

fragment float4 fragmentBaseRendering(FragmentIn in [[ stage_in ]],
                                      texture2d<float, access::sample> texture [[ texture(0) ]],
                                      texture2d_array<float, access::sample> segmentationMask [[ texture(1) ]],
                                      constant SegmentationParams& segmentationParams [[ buffer(0) ]],
                                      constant BBox* bboxes [[ buffer(1) ]],
                                      uint index [[ viewport_array_index ]])
{
  constexpr sampler imageSampler(coord::normalized,
                                 filter::nearest);

  float maxConfidence = 0;
  int classId = -1;
  for (uint i = 0; i < segmentationParams.bboxCount; ++i) {
//    BBox bbox = bboxes[i];
//    if (int(gid.x) > bbox.x && int(gid.x) < (bbox.x + bbox.w)
//        && int(gid.y) > bbox.y && int(gid.y) < (bbox.y + bbox.h))
//    {
//    }
    float maskValue = segmentationMask.sample(imageSampler, in.uv, i).r;
    float sigmoid = 1 / (1 + exp(-maskValue));
    if (maskValue != 0 && sigmoid > segmentationParams.confidence && sigmoid > maxConfidence) {
      maxConfidence = sigmoid;
      classId = i;
    }
  }

  if (classId != -1) {
    return float4(1, 0, 0, 1);
  } else {
    return float4(texture.sample(imageSampler, in.uv).rgb, 1);
  }
}
