#version 460
#pragma shader_stage(compute)

///
///
///

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba32f) restrict readonly  uniform image2D CoefficientImage;
layout(set = 0, binding = 1, rgba32f) restrict writeonly uniform image2D ComponentImage;

const float C[8] = float[8](0.707106781187, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
const float PI = 3.141592653589;

void main()
{
	ivec2 pos = ivec2(gl_LocalInvocationID.xy);
	ivec2 gbl = ivec2(gl_GlobalInvocationID.xy);

	vec4 acc = vec4(0.0);
	for(int i = 0; i < 8; ++i)
	{
		for(int j = 0; j < 8; ++j)
		{
			float a = cos((2.0 * float(pos.x) + 1.0) * PI * j);
			float b = cos((2.0 * float(pos.y) + 1.0) * PI * i);

			acc += C[i] * C[j] * vec4(imageLoad(CoefficientImage, gbl)) * a * b;
		}
	}
	acc *= 0.25;

	/* At this point, the value in the accumulator is in the [-256, 255] range,
	 * and, our output frame expects is expected to contain 32bit floating point
	 * color data, in the [0.0, 1.0] range.
	 *
	 * The last step of the decoding process is to map the former range into the
	 * latter, then, we simply store the result in the target YCbCrA image. */
	acc = clamp((acc + 256.0) / 512.0, 0.0, 1.0);

	imageStore(ComponentImage, gbl, acc);
}
