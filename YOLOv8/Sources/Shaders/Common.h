//
// Created by moonl1ght 27.02.2023.
//

#ifndef Common_h
#define Common_h

#import <simd/simd.h>

typedef enum VertexAttributes {
  kVertexAttributePosition = 0,
  kVertexAttributeUV = 1
} VertexAttributes;

typedef struct {
  matrix_float4x4 scaleMatrix;
} VertexUniforms;

#endif /* Common_h */
