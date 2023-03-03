//
// Created by moonl1ght 27.02.2023.
//

#include <metal_stdlib>
#import "Common.h"

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
