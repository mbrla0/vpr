use std::collections::VecDeque;
use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use crate::decoder::Decoder;
use crate::{Context, DecodeScheduler, DeviceContext};
use ash::vk;
use crate::image::{Frame, Image};
mod syntax;
mod endian;

pub struct ProRes;
impl Decoder for ProRes {
	type SharedState = SharedState;
	type InstanceState = InstanceState;
	type WorkerState = WorkerState;
	type FrameState = FrameState;
	type FrameParam = ();
	type Error = ();

	fn schedule(&self,
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &mut Self::InstanceState,
		frames: &DecodeScheduler<Self>,
		data: &[u8]
	) -> Result<(), Self::Error> {
		todo!()
	}

	fn decode(&self,
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		worker: &mut Self::WorkerState,
		frame: &mut Frame<Self::FrameState>,
		param: Self::FrameParam,
		data: &[u8]
	) {
		todo!()
	}

	fn create_shared_state(context: &DeviceContext) -> Result<Self::SharedState, ()> {
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

	fn create_instance_state(
		context: &DeviceContext,
		shared: &Self::SharedState
	) -> Result<Self::InstanceState, Self::Error> {
		todo!()
	}

	fn create_worker_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState
	) -> Result<Self::WorkerState, Self::Error> {
		todo!()
	}

	fn create_frame_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		worker: &mut Self::WorkerState,
		image: &Image
	) -> Result<Self::FrameState, Self::Error> {
		todo!()
	}

	fn destroy_shared_state(
		context: &DeviceContext,
		shared: &mut Self::SharedState
	) {
		let vk = &context.device;
		unsafe {
			vk.destroy_shader_module(shared.unpack_slices, None);
			vk.destroy_shader_module(shared.idct, None);
			vk.destroy_descriptor_set_layout(shared.image_set_layout, None);
			vk.destroy_descriptor_set_layout(shared.frame_set_layout, None);
		}
	}

	fn destroy_instance_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &mut Self::InstanceState
	) {
		todo!()
	}

	fn destroy_worker_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		worker: &mut Self::WorkerState
	) {
		todo!()
	}

	fn destroy_frame_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		worker: &mut Self::WorkerState,
		frame: &mut Self::FrameState
	) {
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

/// Instances recycling queue-related functions for all descriptor set types
/// the worker state may wish to distinguish between.
macro_rules! instance_descriptor_set_functions {
	($(
		pub unsafe (
			$(#[$getter_meta:meta])*
			fn $getter:ident,
			$(#[$ret_meta:meta])*
			fn $ret:ident
		) => $queue:ident;
	)*) => {$(
		$(#[$getter_meta])*
		pub unsafe fn $getter(
			&mut self,
			context: &DeviceContext,
			shared: &SharedState,
		) -> VkResult<vk::DescriptorSet> {
			let vk = context.device();
			if let Some(buffer) = self.$queue.pop_front() {
				vk.reset_command_buffer(
					buffer,
					vk::CommandBufferResetFlags::empty())?;
				Ok(buffer)
			} else {
				let buffers = vk.allocate_command_buffers(
					&vk::CommandBufferAllocateInfo::builder()
						.command_pool(self.command_buffer_pool)
						.command_buffer_count(1)
						.build())?;
				Ok(buffers[0])
			}
		}

		$(#[$ret_meta])*
		pub unsafe fn $ret(&mut self, set: vk::DescriptorSet) {
			self.$queue.push_back(set)
		}
	)*}
}

struct WorkerState {
	command_buffer_pool: vk::CommandPool,
	descriptor_set_pool: vk::DescriptorPool,
	command_buffer_recycling_queue: VecDeque<vk::CommandBuffer>,
	image_descriptor_set_recycling_queue: VecDeque<vk::DescriptorSet>,
	frame_descriptor_set_recycling_queue: VecDeque<vk::DescriptorSet>,
}
impl WorkerState {
	/// Recycles an already existing command buffer or creates a new one if none
	/// are available and returns it.
	pub unsafe fn get_command_buffer(
		&mut self,
		context: &DeviceContext
	) -> VkResult<vk::CommandBuffer> {
		let vk = context.device();
		if let Some(buffer) = self.command_buffer_recycling_queue.pop_front() {
			vk.reset_command_buffer(
				buffer,
				vk::CommandBufferResetFlags::empty())?;
			Ok(buffer)
		} else {
			let buffers = vk.allocate_command_buffers(
				&vk::CommandBufferAllocateInfo::builder()
					.command_pool(self.command_buffer_pool)
					.command_buffer_count(1)
					.build())?;
			Ok(buffers[0])
		}
	}

	instance_descriptor_set_functions! {
		pub unsafe (
			#[doc = "Returns a new image descriptor set."]
			fn get_image_descriptor_set,
			#[doc = "Recycles an old image descriptor set that's no longer in use."]
			fn ret_image_descriptor_set
		) => image_descriptor_set_recycling_queue;
		pub unsafe (
			#[doc = "Returns a new frame descriptor set for."]
			fn get_image_descriptor_set,
			#[doc = "Recycles an old frame descriptor set that's no longer in use."]
			fn ret_frame_descriptor_set
		) => frame_descriptor_set_recycling_queue;
	}
}

pub struct FrameState {
	bindings: ArrayVec<vk::DescriptorSet, 4>,
	commands: vk::CommandBuffer,
}

mod shaders {
	#[sshdr::include(file = "../../../shaders/prores/UnpackSlices.glsl", stage = "compute")]
	pub static UNPACK_SLICES: &'static [u32];

	#[sshdr::include(file = "../../../shaders/prores/IDCT.glsl", stage = "compute")]
	pub static IDCT: &'static [u32];
}
