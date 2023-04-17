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
                                      texture2d<float, access::sample> texture [[ texture(0) ]])
{
  constexpr sampler imageSampler(coord::normalized,
                                 filter::linear);
  return float4(texture.sample(imageSampler, in.uv).rgb, 1);
}

fragment float4 fragmentSegmentationRendering(FragmentIn in [[ stage_in ]],
                                              texture2d<float, access::sample> texture [[ texture(0) ]],
                                              texture2d_array<float, access::sample> segmentationMask [[ texture(1) ]],
                                              constant SegmentationParams& segmentationParams [[ buffer(0) ]],
                                              constant BBox* bboxes [[ buffer(1) ]],
                                              constant simd_float3* colors [[ buffer(2) ]])
{
  constexpr sampler imageSampler(coord::normalized,
                                 filter::linear);

  float maxConfidence = 0;
  int classId = -1;
  if (!is_null_texture(segmentationMask)) {
    for (uint i = 0; i < segmentationParams.bboxCount; ++i) {
      BBox bbox = bboxes[i];
      int2 point = int2(in.position.xy);
      if (point.x < bbox.x || point.x > (bbox.x + bbox.w)
          || point.y < bbox.y || point.y > (bbox.y + bbox.h))
      {
        continue;
      }
      float maskValue = segmentationMask.sample(imageSampler, in.uv, i).r;
      float sigmoid = 1 / (1 + exp(-maskValue));
      if (maskValue != 0 && sigmoid > segmentationParams.confidence && sigmoid > maxConfidence) {
        maxConfidence = sigmoid;
        classId = bbox.classId;
      }
    }
  }

  float4 textureColor = float4(texture.sample(imageSampler, in.uv).rgb, 1);
  if (classId != -1) {
    float4 color = float4(colors[classId], 1);
    float4 outColor = 0.5 * color + (1 - 0.5) * textureColor;
    return outColor;
  } else {
    return textureColor;
  }
}
