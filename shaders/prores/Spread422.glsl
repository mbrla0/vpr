#version 460
#pragma shader_stage(compute)

/// 4:2:2 Color Spread
///
/// This program takes the YCbCrA image produced by the IDCT procedure on a
/// 4:2:2 chroma subsampled frame and produces the final YCbCrA image. At this
/// stage of the process of decoding ProRes, these images can be divided into
/// multiple 16 pixel vertical strips, where only the first 8 pixels of each
/// strip contain valid color data. We have, therefore, to spread this color
/// data so that it covers the whole strip.
///
/// The spread is done by repeating each color component twice horizontally,
/// so that for any line of Cb or Cr components in the vertical strip that
/// started off as:
///
/// | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | X | X | X | X | X | X | X | X |
///
/// Will, after the spread is complete, look like:
///
/// | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 4 | 5 | 5 | 6 | 6 | 7 | 7 |
///

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 1, rgba32f) restrict readonly  uniform image2D ComponentImage;
layout(set = 0, binding = 2, rgba32f) restrict writeonly uniform image2D FinalImage;

void main()
{
	ivec2 gbl = ivec2(gl_GlobalInvocationID.xy);

	/* Find the positions of all of the working items. */
	int strip = gbl.x >> 3;
	int offset = gbl.x & 7;

	ivec2 colorSourcePos = ivec2((strip << 1) + offset, gbl.y);
	ivec2 leftPos = ivec2((strip << 1) + (offset << 1) + 0, gbl.y);
	ivec2 rightPos = ivec2((strip << 1) + (offset << 1) + 1, gbl.y);

	/* Load the color data from all of our sources. */
	vec4 colorSource = imageLoad(ComponentImage, colorSourcePos);
	vec4 left = imageLoad(ComponentImage, leftPos);
	vec4 right = imageLoad(ComponentImage, rightPos);

	/* Perform the 4:2:2 color spread in the Cb and Cr components. */
	left.yz = colorSource.yz;
	right.yz = colorSource.yz;

	/* Store the data into the target positions in the final image. */
	imageStore(FinalImage, leftPos, left);
	imageStore(FinalImage, rightPos, right);
}
