use std::collections::linked_list::LinkedList;
use std::collections::vec_deque::VecDeque;
use arrayvec::ArrayVec;
use ash::prelude::VkResult;
use crate::decoder::Decoder;
use crate::{VprContext, DecodeScheduler, DeviceContext, Dimensions};

use ash::vk;
use byteorder::{BigEndian, ReadBytesExt};
use crate::image::{Frame, ImageView};

mod parser;
mod endian;
mod macros;

/// An analogue for the `IndexEntry` struct in `UnpackSlices.glsl`.
#[repr(C)]
struct UnpackSlicesIndexEntry {
	/// The offset of the start of the slice.
	offset: u32,
	/// The coded position and size of the slice.
	///
	/// This is a bit-packed structure with the following fields:
	/// - `0..16`: Width in macroblocks.
	/// - `16..30`: Height in macroblocks.
	/// - `30..32`: log2(Size of the slice in macroblocks).
	position: u32,
	/// The size of the compressed slice data, in bytes.
	coded_size: u32,
	/// Padding value.
	///
	/// It is explicitly defined in the shader, so it may have any value. It is polite to leave it
	/// set to zero, however, as removing the field in the shader will restrict it to that value.
	padding0: u32,
}

pub struct ProRes;
impl ProRes {
	/// ProRes frames all have a special bit string that comes right after the frame size.
	///
	/// Checking for the presence of the string lets us check if the data we're reading is probably
	/// in the correct format and correctly aligned, as opposed to being just some random bytes we
	/// happened to receive or land on.
	const FRAME_IDENTIFIER: &'static [u8; 4] = b"icpf";

	/// This is the maximum length of a slice, measured in macroblocks, in log2.
	const MAX_LOG2_MB_SLICE_SIZE: u8 = 3;
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
		frames: &mut DecodeScheduler<Self>,
		data: &[u8]
	) -> Result<usize, Self::Error> {
		if data.len() < 4 {
			/* Not enough data to know how long the next frame is. */
			return Ok(0)
		}

		/* This procedure is gonna be running in a single thread, so we want it to be over as soon
		 * as possible. Just divvy up the frames as quickly as we can so that the workers can deal
		 * with them in parallel. */
		let mut cursor = data;
		let mut offset = 0;
		loop {
			if cursor.len() < 8 { break }
			let len: usize = cursor.read_u32::<BigEndian>().unwrap().try_into().unwrap();
			let sig = cursor.split_array_ref::<4>().0.clone();
			cursor = &cursor[4..];

			/* Reject the data if the signature on it doesn't match the frame identifier. */
			if sig != Self::FRAME_IDENTIFIER {
				return Err(Self::Error::InvalidFrameSignature(sig))
			}

			/* Wait for the full frame to become available before we schedule it. Hopefully the
			 * length we read isn't corrupted or wrong, or we'll be waiting until the end of the
			 * data before we realize there's something wrong. */
			if cursor.len() < len { break }

			/* Before we try to parse anything out, make sure the data is at least long enough that
			 * we can read the fields we're interested in for the scheduling. Right now, we don't
			 * care if the frame structure has enough data to produce an image, we can leave these
			 * more thorough checks to the parallel workers. */
			if cursor.len() < 12 {
				return Err(Self::Error::UnexpectedEndOfFrame)
			}

			/* Skip until the dimensions inside the frame header and read them. We currently ignore
			 * all of the data in the header until this point. */
			cursor = &cursor[8..];

			let mut dimensions = [0u16; 2];
			cursor.read_u16_into::<BigEndian>(&mut dimensions).unwrap();

			let dimensions = Dimensions {
				width: u32::from(dimensions[0]),
				height: u32::from(dimensions[1])
			};

			frames.schedule(offset..offset + len, dimensions, ());
			offset += len;
		}

		Ok(offset)
	}

	fn decode(&self,
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		worker: &mut Self::WorkerState,
		frame: &mut Frame<Self::FrameState>,
		_param: Self::FrameParam,
		data: &[u8]
	) {
		let frame_parser = parser::Frame::new(data).unwrap();

		/* Construct the slice layout of the picture.
		 *
		 * We have to scan through the slices in order to divvy up the work amongst the threads in
		 * the device, with each thread processing a single slice.
		 *
		 * Additionally, the shaders handle the image in macroblock units, but the bitstream simply
		 * lays slices out such that they appear in scan order, from left to right, top to bottom.
		 * This means we have to derive this information from the slices, as well as how many
		 * macroblocks each individual slice corresponds to, before the shader is invoked.
		 */
		let mb_width = ((frame_parser.header().horizontal_size() as u32 + 15) / 16) as u16;
		let mb_height = ((frame_parser.header().horizontal_size() as u32 + 15) / 16) as u16;

		let slice_count_per_row = {
			/* Because slices sizes are guaranteed to be powers of two, and have a maximum size of
			 * eight, we may simply add the unsigned number after bit offset 3 to the number of set
			 * bits below that offset.
			 *
			 * To illustrate why this works, imagine the width of the row as being given by the
			 * following unsigned integer in binary representation:
			 *
			 * |-----------------|-------|
			 * | . . . 1 1 1 1 1 | 1 1 1 |
			 * |--------A--------|---B---|
			 *
			 * Section A is, obviously, the maximum number of 8-sized slices that can be used to
			 * cover the area of a row, as it is the same as the result of the division of the width
			 * by 8. Section B, meanwhile, gives us the exact number of each of the remaining slice
			 * sizes that are needed to fill the rest of the area in the same row, using the fewest
			 * number of slices possible.
			 *
			 * The same logic applies to all possible values for the maximum size of a slice,
			 * provided that the bit offset that separates regions A and B is set appropriately.
			 */
			let div = mb_width >> Self::MAX_LOG2_MB_SLICE_SIZE;
			let rem = mb_width % (1 << Self::MAX_LOG2_MB_SLICE_SIZE);

			div + rem.count_ones()
		};
		let slice_lengths = frame_parser.picture0().read_slice_sizes(mb_height, slice_count_per_row)
			.map(|(x, y, val)|)
			.collect::<>
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

			let spread_pipeline = macros::create_compute_pipeline! {
				device: vk,
				cache: shared.pipeline_cache,

			};

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
	spread_pipeline_layout: vk::PipelineLayout,

	unpack_slices: vk::ShaderModule,
	idct: vk::ShaderModule,
	spread: vk::ShaderModule,

	image_set_layout: vk::DescriptorSetLayout,
	frame_set_layout: vk::DescriptorSetLayout,
}

pub struct InstanceState {
	decoder_state: DecoderState,

	unpack_pipeline: vk::Pipeline,
	idct_pipeline: vk::Pipeline,
	spread_pipeline: vk::Pipeline,
}

enum DecoderState {

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
			fn ret_image_descriptor_set
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
	#[error("frame contains invalid identifier: {0:?}")]
	InvalidFrameSignature([u8; 4]),
	#[error("frame data ends sooner than expected")]
	UnexpectedEndOfFrame
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
