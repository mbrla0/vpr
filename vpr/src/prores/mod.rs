use crate::decode::Decoder;
use crate::{Context, DecodeScheduler, VulkanContext};
use ash::vk;
use crate::image::{Frame, Image};
mod syntax;
mod endian;

pub struct ProRes;
impl Decoder for ProRes {
	type SharedState = SharedState;
	type InstanceState = InstanceState;
	type FrameState = ();
	type Error = ();

	fn schedule(&self, context: &Context, shared: &Self::SharedState, instance: &mut Self::InstanceState, frames: &DecodeScheduler<Self>, data: &[u8]) {
		todo!()
	}

	fn decode(&self, context: &Context, shared: &Self::SharedState, instance: &Self::InstanceState, frame: &Frame<Self::FrameState>, data: &[u8]) {
		todo!()
	}

	fn create_shared_state(context: &VulkanContext) -> Result<Self::SharedState, ()> {
		let vk = &context.device;
		unsafe {
			let unpack_slices = vk.create_shader_module(
				&vk::ShaderModuleCreateInfo::builder()
					.code(shaders::UNPACK_SLICES)
					.build(),
				None)?;
			let idct = vk.create_shader_module(
				&vk::ShaderModuleCreateInfo::builder()
					.code(shaders::IDCT)
					.build(),
				None)?;


			let frame_set_layout = vk.create_descriptor_set_layout(
				&vk::DescriptorSetLayoutCreateInfo::builder()
					.bindings(&[
						vk::DescriptorSetLayoutBinding::builder()
							.binding(0)
							.descriptor_count(1)
							.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
							.build(),
						vk::DescriptorSetLayoutBinding::builder()
							.binding(1)
							.descriptor_count(1)
							.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
							.build(),
					])
					.build(),
				None)?;

			let image_set_layout = vk.create_descriptor_set_layout(
				&vk::DescriptorSetLayoutCreateInfo::builder()
					.bindings(&[
						vk::DescriptorSetLayoutBinding::builder()
							.binding(0)
							.descriptor_count(1)
							.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
							.build(),
						vk::DescriptorSetLayoutBinding::builder()
							.binding(1)
							.descriptor_count(1)
							.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
							.build(),
					])
					.build(),
				None)?;

			let unpack_pipeline_layout = vk.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[
						frame_set_layout,
						image_set_layout,
					])
					.push_constant_ranges(&[
						vk::PushConstantRange::builder()
							.stage_flags(vk::ShaderStageFlags::COMPUTE)
							.offset(0)
							.size(4)
							.build(),
						vk::PushConstantRange::builder()
							.stage_flags(vk::ShaderStageFlags::COMPUTE)
							.offset(4)
							.size(4)
							.build(),
						vk::PushConstantRange::builder()
							.stage_flags(vk::ShaderStageFlags::COMPUTE)
							.offset(8)
							.size(4)
							.build(),
						vk::PushConstantRange::builder()
							.stage_flags(vk::ShaderStageFlags::COMPUTE)
							.offset(12)
							.size(4)
							.build(),
					])
					.build(),
				None)?;

			let idct_pipeline_layout = vk.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[
						image_set_layout
					])
					.build(),
				None)?;

			Ok(Self::SharedState {
				unpack_pipeline_layout,
				idct_pipeline_layout,
				unpack_slices,
				idct,
				image_set_layout,
				frame_set_layout
			})
		}
	}

	fn create_instance_state(context: &VulkanContext) -> Result<Self::InstanceState, ()> {
		let vk = &context.device;
		unsafe {

		}
	}

	fn create_frame_state(context: &Context, shared: &Self::SharedState, instance: &Self::InstanceState, image: &Image) -> Result<Self::FrameState, Self::Error> {
		todo!()
	}
	fn destroy_shared_state(context: &VulkanContext, shared: &mut Self::SharedState) {
		let vk = &context.device;
		unsafe {
			vk.destroy_shader_module(shared.unpack_slices, None);
			vk.destroy_shader_module(shared.idct, None);
			vk.destroy_descriptor_set_layout(shared.image_set_layout, None);
			vk.destroy_descriptor_set_layout(shared.frame_set_layout, None);
		}
	}
	fn destroy_instance_state(context: &VulkanContext, instance: &mut Self::InstanceState) {
		let vk = &context.device;
		unsafe {
			vk.free_command_buffers()
		}
	}
	fn destroy_frame_state(context: &Context, shared: &Self::SharedState, instance: &Self::InstanceState, frame: &mut Self::FrameState) {
		todo!()
	}
}

pub struct SharedState {
	unpack_pipeline_layout: vk::PipelineLayout,
	idct_pipeline_layout: vk::PipelineLayout,

	unpack_slices: vk::ShaderModule,
	idct: vk::ShaderModule,

	image_set_layout: vk::DescriptorSetLayout,
	frame_set_layout: vk::DescriptorSetLayout,
}

pub struct InstanceState {

}

pub struct FrameState {
	command_buffer:
}

mod shaders {
	#[sshdr::include(file = "../../../shaders/prores/UnpackSlices.glsl", stage = "compute")]
	pub static UNPACK_SLICES: &'static [u32];

	#[sshdr::include(file = "../../../shaders/prores/IDCT.glsl", stage = "compute")]
	pub static IDCT: &'static [u32];
}
