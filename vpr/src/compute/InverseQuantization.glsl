#version 460
#pragma shader_stage(compute)

#define COLOR_MODE_LUMA 0
#define COLOR_MODE_CHROMA 1

layout(constant_id = 0) const int cColorMode = COLOR_MODE_LUMA;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) restrict readonly buffer _QuantizationTable
{
	uint gYQuantizationMatrix[16];
	uint gCQuantizationMatrix[16];
};

#define QUANTIZATION_Y(i, j) ((gYQuantizationMatrix[i + (j >> 2U)] >> ((j & 3U) * 8)) & 0xFFU)
#define QUANTIZATION_C(i, j) ((gCQuantizationMatrix[i + (j >> 2U)] >> ((j & 3U) * 8)) & 0xFFU)

void main()
{
	
}