#version 460
#pragma shader_stage(compute)

///
///
///

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 1, rgba32f) restrict readonly  uniform image2D ComponentImage;
layout(set = 0, binding = 2, rgba32f) restrict writeonly uniform image2D FinalImage;

void main()
{
	ivec2 gbl = ivec2(gl_GlobalInvocationID.xy);

	vec4 sourceL
	vec4 sourceL = imageLoad(ComponentImage, gbl * ivec2(8, 0));
	vec4 sourceR = imageLoad(ComponentImage, gbl * ivec2(8, 0));

	imageStore(ComponentImage, gbl, acc);
}
