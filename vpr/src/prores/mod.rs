use std::collections::linked_list::LinkedList;
use std::collections::vec_deque::VecDeque;
use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use crate::decoder::Decoder;
use crate::{VprContext, DecodeScheduler, DeviceContext};

use ash::vk;
use crate::image::{Frame, ImageView};

mod syntax;
mod endian;
mod macros;

pub struct ProRes;
impl ProRes {
}
impl Decoder for ProRes {
	type SharedState = SharedState;
	type InstanceState = InstanceState;
	type WorkerState = WorkerState;
	type FrameState = FrameState;
	type FrameParam = ();
	type Error = Error;

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

	fn create_shared_state(context: &DeviceContext) -> Result<Self::SharedState, Self::Error> {
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
					.build(),
				None)?;

			let idct_pipeline_layout = vk.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[
						image_set_layout
					])
					.build(),
				None)?;

			/* TODO: Maybe there ought to be a storage class for cache persistence. */
			let pipeline_cache = vk.create_pipeline_cache(
				&vk::PipelineCacheCreateInfo::builder()
					.build(),
				None)?;

			/* Pick a queue family. */
			let queue_family_index = context.queue_family_properties()
				.into_iter()
				.enumerate()
				.find(|(_, properties)| {
					properties.queue_flags.contains(
						vk::QueueFlags::COMPUTE
							| vk::QueueFlags::TRANSFER)
				})
				.map(|(i, _)| i)
				.unwrap() as u32;

			Ok(SharedState {
				pipeline_cache,
				queue_family_index,
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
		let endianness = endian::host_endianness()?;
		let vk = context.device();
		unsafe {
			let unpack_pipeline = macros::create_compute_pipeline! {
				device: vk,
				cache: shared.pipeline_cache,
				layout: shared.unpack_pipeline_layout,
				module: shared.unpack_slices,
				entry: "main",
				specialization: [
					/* int cLittleEndian */
					{ index: 0, offset: 0x00, data: endianness.flag() },
					/* int cSubsamplingMode */
					{ index: 1, offset: 0x04, data: &[] },
					/* int cAlphaFormat */
					{ index: 2, offset: 0x08, data: &[] },
					/* int cScanningMode */
					{ index: 3, offset: 0x0c, data: &[0] }
				]
			}?;

			let idct_pipeline = macros::create_compute_pipeline! {
				device: vk,
				cache: shared.pipeline_cache,
				layout: shared.idct_pipeline_layout,
				module: shared.idct,
				entry: "main",
				specialization: []
			}?;

			Ok(InstanceState {
				unpack_pipeline,
				idct_pipeline
			})
		}
	}

	fn create_worker_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState
	) -> Result<Self::WorkerState, Self::Error> {
		let vk = context.device();
		unsafe {
			let command_buffer_pool = vk.create_command_pool(
				&vk::CommandPoolCreateInfo::builder()
					.queue_family_index(shared.queue_family_index)
					.build(),
				None)?;

			let mut state = WorkerState {
				command_buffer_pool,
				descriptor_set_pool: Default::default(),
				command_buffer_recycling_queue: Default::default(),
				image_descriptor_set_recycling_queue: Default::default(),
				frame_descriptor_set_recycling_queue: Default::default(),
			};
			let _ = state.new_descriptor_pool(context)?;
			Ok(state)
		}
	}

	fn create_frame_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		worker: &mut Self::WorkerState,
		image: &ImageView
	) -> Result<Self::FrameState, Self::Error> {

		let vk = context.device();
		unsafe {
			let frame_bind = worker.get_frame_descriptor_set(context, shared)?;
			let image_bind = worker.get_image_descriptor_set(context, shared)?;

			vk.update_descriptor_sets(
				&[
					vk::WriteDescriptorSet::builder()
						.dst_set(image_bind)
						.dst_binding(0)
						.dst_array_element(0)
						.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
						.image_info(&[
							vk::DescriptorImageInfo::builder()
								.image_layout(vk::ImageLayout::GENERAL)
								.image_view()
								.build(),
							vk::DescriptorImageInfo::builder()
								.image_layout(vk::ImageLayout::GENERAL)
								.image_view(image.raw_handle())
								.build()
						])
						.build()
				],
				&[])?;

			let commands = worker.get_command_buffer(context)?;
			vk.reset_command_buffer(
				commands,
				vk::CommandBufferResetFlags::empty())?;

			vk.cmd_bind_pipeline(
				commands,
				vk::PipelineBindPoint::COMPUTE,
				instance.unpack_pipeline);

			vk.cmd_bind_descriptor_sets(
				commands,
				vk::PipelineBindPoint::COMPUTE,
				shared.unpack_pipeline_layout,
				0,
				&[frame_bind],
				&[0]);

			vk.cmd_dispatch()


			Ok(FrameState {
				frame_bind,
				image_bind,
				commands,
			})
		}
	}

	fn destroy_shared_state(
		context: &DeviceContext,
		shared: &mut Self::SharedState
	) {
		let vk = &context.device;
		unsafe {
			vk.destroy_pipeline_cache(shared.pipeline_cache, None);
			vk.destroy_shader_module(shared.unpack_slices, None);
			vk.destroy_shader_module(shared.idct, None);
			vk.destroy_pipeline_layout(shared.unpack_pipeline_layout, None);
			vk.destroy_pipeline_layout(shared.idct_pipeline_layout, None);
			vk.destroy_descriptor_set_layout(shared.image_set_layout, None);
			vk.destroy_descriptor_set_layout(shared.frame_set_layout, None);
		}
	}

	fn destroy_instance_state(
		context: &DeviceContext,
		_shared: &Self::SharedState,
		instance: &mut Self::InstanceState
	) {
		let vk = context.device();
		unsafe {
			vk.destroy_pipeline(instance.unpack_pipeline, None);
			vk.destroy_pipeline(instance.idct_pipeline, None);
		}
	}

	fn destroy_worker_state(
		context: &DeviceContext,
		_shared: &Self::SharedState,
		_instance: &Self::InstanceState,
		worker: &mut Self::WorkerState
	) {
		let vk = context.device();
		unsafe {
			vk.destroy_command_pool(worker.command_buffer_pool, None);
			for i in worker.descriptor_set_pool {
				vk.destroy_descriptor_pool(i, None);
			}
		}
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
	pipeline_cache: vk::PipelineCache,
	queue_family_index: u32,

	unpack_pipeline_layout: vk::PipelineLayout,
	idct_pipeline_layout: vk::PipelineLayout,

	unpack_slices: vk::ShaderModule,
	idct: vk::ShaderModule,

	image_set_layout: vk::DescriptorSetLayout,
	frame_set_layout: vk::DescriptorSetLayout,
}

pub struct InstanceState {
	unpack_pipeline: vk::Pipeline,
	idct_pipeline: vk::Pipeline,
}

struct WorkerState {
	command_buffer_pool: vk::CommandPool,
	descriptor_set_pool: LinkedList<vk::DescriptorPool>,
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

	unsafe fn new_descriptor_pool(
		&mut self,
		context: &DeviceContext
	) -> VkResult<vk::DescriptorPool> {

		/// Maximum number of instances that will fit in a single pool.
		///
		/// TODO: Having a fixed number of instances per descriptor pool is wasteful.
		const INSTANCES: u32 = 64;

		let vk = context.device();
		let pool = unsafe {
			vk.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo::builder()
					.max_sets(INSTANCES * 2)
					.pool_sizes(&[
						vk::DescriptorPoolSize::builder()
							.ty(vk::DescriptorType::STORAGE_BUFFER)
							.descriptor_count(INSTANCES * 2)
							.build(),
						vk::DescriptorPoolSize::builder()
							.ty(vk::DescriptorType::STORAGE_IMAGE)
							.descriptor_count(INSTANCES * 2)
							.build(),
					])
					.build(),
				None)?
		};
		self.descriptor_set_pool.push_back(pool);
		Ok(pool)
	}

	pub unsafe fn allocate_descriptor_set(
		&mut self,
		context: &DeviceContext,
		layout: vk::DescriptorSetLayout,
	) -> VkResult<vk::DescriptorSet> {
		let vk = context.device();
		let attempt = |pool| unsafe {
			vk.allocate_descriptor_sets(
				&vk::DescriptorSetAllocateInfo::builder()
					.descriptor_pool(pool)
					.set_layouts(&[layout])
					.build())
		};

		match attempt(*self.descriptor_set_pool.back().unwrap()) {
			Ok(sets) => Ok(sets[0]),
			Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)
			| Err(vk::Result::ERROR_FRAGMENTED_POOL) =>
				Ok(attempt(self.new_descriptor_pool(context)?)?[0])
		}
	}

	macros::instance_descriptor_set_functions! {
		pub unsafe (
			#[doc = "Returns a new image descriptor set."]
			fn get_image_descriptor_set,
			#[doc = "Recycles an old image descriptor set that's no longer in use."]
			fn get_image_descriptor_set
		) => image_descriptor_set_recycling_queue;
		pub unsafe (
			#[doc = "Returns a new frame descriptor set for."]
			fn get_frame_descriptor_set,
			#[doc = "Recycles an old frame descriptor set that's no longer in use."]
			fn ret_frame_descriptor_set
		) => frame_descriptor_set_recycling_queue;
	}
}

pub struct FrameState {
	frame_bind: vk::DescriptorSet,
	image_bind: vk::DescriptorSet,
	commands: vk::CommandBuffer,

}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("vulkan error: {0:?}")]
	Vulkan(vk::Result),
	#[error("system endianness is not supported")]
	UnsupportedEndianness,
}
impl From<vk::Result> for Error {
	fn from(value: vk::Result) -> Self {
		Self::Vulkan(value)
	}
}
impl From<endian::UnsupportedEndianness> for Error {
	fn from(_: endian::UnsupportedEndianness) -> Self {
		Self::UnsupportedEndianness
	}
}

mod shaders {
	#[sshdr::include(file = "../../../shaders/prores/UnpackSlices.glsl", stage = "compute")]
	pub static UNPACK_SLICES: &'static [u32];

	#[sshdr::include(file = "../../../shaders/prores/IDCT.glsl", stage = "compute")]
	pub static IDCT: &'static [u32];
}
