#version 460
#pragma shader_stage(compute)

///
///
///

struct IndexEntry
{
	uint offset;
	uint position;
};

layout(bind = 0, location = 0, std430) restrict readonly buffer _InputSliceIndex
{
	IndexEntry elements[];
} InputSliceIndex;
layout(bind = 0, location = 1, std430) restrict readonly buffer _InputData
{
	uint blob[];
} InputData;

layout(bind = 1, location = 0, rgba32i) restrict writeonly uniform image2D CoefficientImage;

const uint BLOCK_SCAN_PROGRESSIVE[64] = uint[64](
	 0,  1,  4,  5, 16, 17, 21, 22,
	 2,  3,  6,  7, 18, 20, 23, 28,
	 8,  9, 12, 13, 19, 24, 27, 29,
	10, 11, 14, 15, 25, 26, 30, 31,
	32, 33, 37, 38, 45, 46, 53, 54,
	34, 36, 39, 44, 47, 52, 55, 60,
	35, 40, 43, 48, 51, 56, 59, 61,
	41, 42, 49, 50, 57, 58, 62, 63
);

const uint BLOCK_SCAN_INTERLACED[64] = uint[64](
	 0,  2,  8, 10, 32, 34, 35, 48,
	 1,  3,  9, 11, 33, 36, 40, 42,
	 4,  6, 12, 14, 37, 39, 43, 49,
	 5,  7, 13, 15, 38, 44, 48, 50,
	16, 18, 19, 25, 45, 47, 51, 57,
	17, 20, 24, 26, 46, 52, 56, 58,
	21, 23, 27, 30, 53, 55, 59, 62,
	22, 28, 29, 31, 54, 60, 61, 63
);

void main()
{
	uint index = gl_GlobalInvocationID.x;

	uvec2 slice_mb_offset = uvec2(
		(InputSliceIndex.elements[index].position & 0x00007fff) >> 0,
		(InputSliceIndex.elements[index].position & 0x3fff8000) >> 15);
	uint log2_slice_len_mb = (InputSliceIndex.elements[index].position & 0xc0000000) >> 30;

	uint word_offset = InputSliceIndex.elements[index].offset >> 2;
	uint byte_offset = IndexSliceIndex.elements[index].offset & 3;


}
