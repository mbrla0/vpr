use std::collections::linked_list::LinkedList;
use std::collections::vec_deque::VecDeque;
use std::intrinsics::unlikely;
use std::ptr::NonNull;
use ash::prelude::VkResult;
use crate::decoder::Decoder;
use crate::{DecodeScheduler, DeviceContext, Dimensions};

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
	/// - `16..32`: Height in macroblocks.
	position: u32,
	/// The size of the compressed slice data, in bytes.
	coded_size: u32,
	/// Coded miscellaneous data.
	///
	/// This is a bit-packed structure with the following fields:
	/// - `0..2`: log2(Number of macroblocks in the slice).
	coded0: u32,
}
impl UnpackSlicesIndexEntry {
	pub fn encode(&self, target: &mut [u8; 16]) {
		target[0..4].copy_from_slice(&self.offset.to_ne_bytes());
		target[4..8].copy_from_slice(&self.position.to_ne_bytes());
		target[8..12].copy_from_slice(&self.coded_size.to_ne_bytes());
		target[12..16].copy_from_slice(&self.coded0.to_ne_bytes());
	}
}

/// An analogue for the quantization matrix data in `UnpackSlices.glsl`
#[repr(C)]
struct UnpackSlicesQuantizationMatrices {
	luma_quantization_matrix: [u8; 64],
	chroma_quantization_matrix: [u8; 64]
}
impl UnpackSlicesQuantizationMatrices {
	pub fn encode(&self, target: &mut [u8; 128]) {
		target[0..64].copy_from_slice(&self.luma_quantization_matrix);
		target[64..128].copy_from_slice(&self.chroma_quantization_matrix);
	}
}

/// Iterator that yields all of the elements in a prefix accumulation.
struct PrefixAccumulate<I: Iterator, F> {
	iter: I,
	acc: Option<I::Item>,
	map: F,
}
impl<I: Iterator, F> Iterator for PrefixAccumulate<I, F>
	where I::Item: Clone,
		  F: FnMut(Option<I::Item>, I::Item) -> I::Item {

	type Item = I::Item;

	fn next(&mut self) -> Option<Self::Item> {
		let next = self.iter.next()?;
		let result = Some(match self.acc.take() {
			None => next,
			Some(prev) => (self.map)(prev, next)
		});

		self.acc = result.clone();
		result
	}
}
trait PrefixAccumulateIterExt: Iterator {
	fn prefix_acc<F>(self, map: F) -> PrefixAccumulate<Self, F>
		where F: FnMut(Option<Self::Item>, Self::Item) -> Self::Item {
		PrefixAccumulate {
			iter: self,
			acc: None,
			map,
		}
	}
}
impl<I: Iterator> PrefixAccumulateIterExt for I {}

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

	/// Default luma quantization matrix.
	///
	/// When a frame does not specify its own Y quantization matrix, this one should be used,
	/// instead. When the CbCr quantization matrix is not specified, it should be the same as the
	/// one used for the Y component.
	const DEFAULT_LUMA_QUANTIZATION_MATRIX: [u8; 64] = [4; 64];

	/// The maximum number of descriptor sets that can be used by a single worker.
	///
	/// Currently, each worker has its own descriptor pool, and this value together with the ones
	/// like it control how many resources each worker may use. Additionally, the number of
	/// descriptors and sets used by a single worker is `O(1)`, and, therefore, the number in this
	/// constant should be the exact maximum.
	///
	/// If descriptor set allocation is failing for a worker, you should either check again that you
	/// haven't missed any descriptor sets or that any changes you have made have it so that the
	/// upper bound on the number of descriptors isn't constant anymore.
	const MAX_WORKER_DESCRIPTOR_SETS: u32 = 2;

	/// The maximum number of storage buffer descriptors that can be used by a single worker.
	///
	/// See [`Self::MAX_WORKER_DESCRIPTOR_SETS`] for more information.
	const MAX_WORKER_STORAGE_BUFFER_DESCRIPTORS: u32 = 2;

	/// The maximum number of storage image descriptors that can be used by a single worker.
	///
	/// See [`Self::MAX_WORKER_DESCRIPTOR_SETS`] for more information.
	const MAX_WORKER_STORAGE_IMAGE_DESCRIPTORS: u32 = 2;
}
impl Decoder for ProRes {
	type SharedState = SharedState;
	type InstanceState = ();
	type WorkerState = WorkerState;
	type FrameState = ();
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
	) -> Result<(), Self::Error> {

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
		let slice_count = vk::DeviceSize::from(slice_count_per_row * mb_height);

		let index_size =
			(std::mem::size_of::<UnpackSlicesQuantizationMatrices>() as vk::DeviceSize)
				.checked_add((std::mem::size_of::<UnpackSlicesIndexEntry>() as vk::DeviceSize)
					.checked_mul(slice_count)
					.unwrap())
				.unwrap();
		let picture_data_size = frame_parser.picture0()
			.read_slice_sizes(mb_height, slice_count_per_row)
			.map(|(_x, _y, size)| vk::DeviceSize::from(size))
			.reduce(|size0, size1| size0.checked_add(size1).unwrap())
			.unwrap();

		/* Upload the picture data to the device. */
		let mut must_invalidate_command_buffers = false;
		if !worker.upload_memory_large_enough(index_size, picture_data_size) {
			unsafe {
				worker.realloc_upload_memory(context, shared, index_size, picture_data_size);
			}
			must_invalidate_command_buffers = true;
		}
		let upload_memory = worker.upload_memory();
		let upload_memory_map = unsafe {
			let i: isize = upload_memory.allocation_size.try_into().unwrap();
			std::slice::from_raw_parts_mut(
				upload_memory.map.as_ptr(),
				i as usize
			)
		};
		let mut cursor = upload_memory_map;

		let luma_quantization_matrix = frame_parser.header().luma_quantization_matrix()
			.unwrap_or(&Self::DEFAULT_LUMA_QUANTIZATION_MATRIX);
		let chroma_quantization_matrix = frame_parser.header().chroma_quantization_matrix()
			.unwrap_or(luma_quantization_matrix);
		UnpackSlicesQuantizationMatrices {
			luma_quantization_matrix: *luma_quantization_matrix,
			chroma_quantization_matrix: *chroma_quantization_matrix,
		}.encode(&mut cursor[..128].try_into().unwrap());
		cursor = &mut cursor[128..];

		let mut last_offset = 0;
		let mut last_row = 0;
		let mut last_x = 0;
		let mut special_slice_index = 0;
		for (slice_no, row, len) in frame_parser.picture0().read_slice_sizes(mb_height, slice_count_per_row) {
			if last_row != row {
				last_x = 0;
				last_row = row;
				special_slice_index = 0;
			}

			/* We can assume that, until the last MAX_LOG2_MB_SLICE_SIZE slices in a row, all of the
			 * slices will have the maximum allowed size, which allows us to skip the somewhat more
			 * involved size calculation for the vast majority of the slices in a picture. */
			let slice_from_end = slice_no - slice_count_per_row - 1;
			let log2_slice_size = if unlikely(slice_from_end < u16::from(Self::MAX_LOG2_MB_SLICE_SIZE)) {
				let mask = 1 << (Self::MAX_LOG2_MB_SLICE_SIZE - special_slice_index) - 1;
				let zeroes = (mb_width & mask).trailing_zeros();

				/* We already know this slice exists. So if we don't find any set bits in the mask
				 * it's either because we're either checking against the wrong size of integer, or
				 * because our bit operations are wrong. Regardless, it's a bug. */
				debug_assert!(zeroes < u16::BITS);

				let size = (u16::BITS - zeroes) as u8;
				special_slice_index += 1;
				size
			} else { Self::MAX_LOG2_MB_SLICE_SIZE };

			UnpackSlicesIndexEntry {
				offset: last_offset,
				position: u32::from(row) << 16 | u32::from(last_x),
				coded_size: 0,
				coded0: u32::from(log2_slice_size),
			}.encode(&mut cursor[..16].try_into().unwrap());
			cursor = &mut cursor[..16];

			last_x = last_x + 1u16 << log2_slice_size as u16;
		}
		unsafe {
			context.device().flush_mapped_memory_ranges(&[vk::MappedMemoryRange::builder()
				.offset(0)
				.size(upload_memory.allocation_size)
				.memory(upload_memory.allocation)
				.build()])
				.unwrap();
		}

		/* Update the internal image buffers so that we can be sure they'll be able to contain the
		 * data for this frame. Additionally, we have to make sure that the size of the working set
		 * of images is rounded up to fit the next whole number of macroblocks, as is required by
		 * ProRes. */

		let coefficient_image_extent = vk::Extent2D {
			width: u32::from(mb_width) * 16,
			height: u32::from(mb_height) * 16,
		};
		let component_image_extent = coefficient_image_extent;

		if !worker.on_device_memory_large_enough(coefficient_image_extent, component_image_extent) {
			unsafe {
				worker.realloc_on_device_memory(
					context,
					shared,
					coefficient_image_extent,
					component_image_extent
				)
			}
			must_invalidate_command_buffers = true;
		}

		/* Dispatch. */
		if must_invalidate_command_buffers {
			worker.invalidate_command_buffers(context);
		}
		let command_buffer = unsafe {
			worker.command_buffer(
				context,
				shared,
				frame,
				FrameLayout {
					mb_width,
					mb_height,
					slice_count_per_row,
				},
				ScanBehavior::Progressive,
				SubsamplingFormat::Subsampling422)
		}.unwrap();

		Ok(())
	}

	fn create_shared_state(context: &DeviceContext) -> Result<Self::SharedState, Self::Error> {
		let vk = context.device();
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
			let spread_422 = vk.create_shader_module(
				&vk::ShaderModuleCreateInfo::builder()
					.code(shaders::SPREAD_422)
					.build(),
				None)?;

			let memory_properties = context.instance()
				.get_physical_device_memory_properties(*context.physical_device());

			let input_data_set_layout = vk.create_descriptor_set_layout(
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

			let working_data_set_layout = vk.create_descriptor_set_layout(
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
						input_data_set_layout,
						working_data_set_layout,
					])
					.build(),
				None)?;

			let idct_pipeline_layout = vk.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[
						working_data_set_layout
					])
					.build(),
				None)?;

			let spread_422_pipeline_layout = vk.create_pipeline_layout(
				&vk::PipelineLayoutCreateInfo::builder()
					.set_layouts(&[
						working_data_set_layout,
						context.frame_descriptor_set_layout()
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

			/* Create all of the pipelines. */
			let endianness = endian::host_endianness()?;
			let unpack_pipeline = macros::create_compute_pipeline! {
				device: vk,
				cache: pipeline_cache,
				layout: unpack_pipeline_layout,
				module: unpack_slices,
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
				cache: pipeline_cache,
				layout: idct_pipeline_layout,
				module: idct,
				entry: "main",
				specialization: []
			}?;

			let spread_422_pipeline = macros::create_compute_pipeline! {
				device: vk,
				cache: pipeline_cache,
				layout: spread_422_pipeline_layout,
				module: spread_422,
				entry: "main",
				specialization: []
			}?;

			Ok(SharedState {
				pipeline_cache,
				queue_family_index,
				memory_properties,
				unpack_pipeline_layout,
				idct_pipeline_layout,
				spread_422_pipeline_layout,
				unpack_pipeline,
				idct_pipeline,
				spread_422_pipeline,
				unpack_slices,
				idct,
				spread_422,
				working_data_set_layout,
				input_data_set_layout
			})
		}
	}

	fn create_instance_state(
		_context: &DeviceContext,
		_shared: &Self::SharedState
	) -> Result<Self::InstanceState, Self::Error> {
		Ok(())
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

			let descriptor_pool = vk.create_descriptor_pool(
				&vk::DescriptorPoolCreateInfo::builder()
					.flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
					.pool_sizes(&[
						vk::DescriptorPoolSize::builder()
							.ty(vk::DescriptorType::STORAGE_BUFFER)
							.descriptor_count(Self::MAX_WORKER_STORAGE_BUFFER_DESCRIPTORS)
							.build(),
						vk::DescriptorPoolSize::builder()
							.ty(vk::DescriptorType::STORAGE_IMAGE)
							.descriptor_count(Self::MAX_WORKER_STORAGE_IMAGE_DESCRIPTORS)
							.build()
					])
					.max_sets(Self::MAX_WORKER_DESCRIPTOR_SETS), None)?;

			let mut state = WorkerState {
				command_pool: command_buffer_pool,
				descriptor_pool,
				leaked_command_buffers: 0,
				progressive_422_command_buffer: None,
				upload_memory: None,
				on_device_memory: None,
			};
			let _ = state.new_descriptor_pool(context)?;
			Ok(state)
		}
	}

	fn create_frame_state(
		_context: &DeviceContext,
		_shared: &Self::SharedState,
		_instance: &Self::InstanceState,
		_worker: &mut Self::WorkerState,
		_image: &ImageView
	) -> Result<Self::FrameState, Self::Error> {
		Ok(())
	}

	fn destroy_shared_state(
		context: &DeviceContext,
		shared: &mut Self::SharedState
	) {
		let vk = context.device();
		unsafe {
			vk.destroy_pipeline_cache(shared.pipeline_cache, None);
			vk.destroy_pipeline(shared.unpack_pipeline, None);
			vk.destroy_pipeline(shared.idct_pipeline, None);
			vk.destroy_pipeline(shared.spread_422_pipeline, None);
			vk.destroy_shader_module(shared.unpack_slices, None);
			vk.destroy_shader_module(shared.idct, None);
			vk.destroy_shader_module(shared.spread_422, None);
			vk.destroy_pipeline_layout(shared.unpack_pipeline_layout, None);
			vk.destroy_pipeline_layout(shared.idct_pipeline_layout, None);
			vk.destroy_pipeline_layout(shared.spread_422_pipeline_layout, None);
			vk.destroy_descriptor_set_layout(shared.working_data_set_layout, None);
			vk.destroy_descriptor_set_layout(shared.input_data_set_layout, None);
		}
	}

	fn destroy_instance_state(
		_context: &DeviceContext,
		_shared: &Self::SharedState,
		_instance: &mut Self::InstanceState
	) {}

	fn destroy_worker_state(
		context: &DeviceContext,
		_shared: &Self::SharedState,
		_instance: &Self::InstanceState,
		worker: &mut Self::WorkerState
	) {
		let vk = context.device();
		unsafe {
			vk.destroy_command_pool(worker.command_pool, None);
			for i in worker.descriptor_pool {
				vk.destroy_descriptor_pool(i, None);
			}
		}
	}

	fn destroy_frame_state(
		_context: &DeviceContext,
		_shared: &Self::SharedState,
		_instance: &Self::InstanceState,
		_worker: &mut Self::WorkerState,
		_frame: &mut Self::FrameState
	) {}
}

pub struct SharedState {
	pipeline_cache: vk::PipelineCache,
	queue_family_index: u32,

	memory_properties: vk::PhysicalDeviceMemoryProperties,

	unpack_pipeline_layout: vk::PipelineLayout,
	idct_pipeline_layout: vk::PipelineLayout,
	spread_422_pipeline_layout: vk::PipelineLayout,

	unpack_pipeline: vk::Pipeline,
	idct_pipeline: vk::Pipeline,
	spread_422_pipeline: vk::Pipeline,

	unpack_slices: vk::ShaderModule,
	idct: vk::ShaderModule,
	spread_422: vk::ShaderModule,

	working_data_set_layout: vk::DescriptorSetLayout,
	input_data_set_layout: vk::DescriptorSetLayout,
}

struct UploadMemory {
	index_buffer_offset: vk::DeviceSize,
	index_buffer_size: vk::DeviceSize,
	index_buffer: vk::Buffer,

	picture_data_buffer_offset: vk::DeviceSize,
	picture_data_buffer_size: vk::DeviceSize,
	picture_data_buffer: vk::Buffer,

	descriptor_set: vk::DescriptorSet,

	allocation_size: vk::DeviceSize,
	allocation: vk::DeviceMemory,
	map: NonNull<u8>
}

struct OnDeviceMemory {
	coefficient_image_extent: vk::Extent2D,
	coefficient_image: vk::Image,

	component_image_extent: vk::Extent2D,
	component_image: vk::Image,

	descriptor_set: vk::DescriptorSet,

	allocation_size: vk::DeviceSize,
	allocation: vk::DeviceMemory,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum SubsamplingFormat {
	Subsampling444,
	Subsampling422
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum ScanBehavior {
	Progressive,
	InterlacedFirstPicture,
	InterlacedSecondPicture,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct FrameLayout {
	/// Number of macroblocks in the horizontal direction.
	mb_width: u16,
	/// Number of macroblocks in the vertical direction.
	mb_height: u16,
	/// Number of slices in every row of macroblocks.
	slice_count_per_row: u16,
}

struct WorkerState {
	command_pool: vk::CommandPool,
	descriptor_pool: vk::DescriptorPool,

	leaked_command_buffers: u8,
	progressive_422_command_buffer: Option<(FrameLayout, vk::CommandBuffer)>,

	upload_memory: Option<UploadMemory>,
	on_device_memory: Option<OnDeviceMemory>,
}
impl WorkerState {
	/// The maximum number of command buffers we are allowed to leak before an invalidation is
	/// forced to happen.
	const MAX_LEAKED_COMMAND_BUFFERS: u8 = 8;

	pub fn upload_memory(&self) -> &UploadMemory {
		self.upload_memory.as_ref().unwrap()
	}

	/// Signals that all command buffers that were previously recorded can't be used again and must
	/// be recreated.
	///
	/// This will often be necessary as a result of the scheduler and decoder dispatch procedures
	/// recreating resources to fit new constraints or because things were moved around in memory.
	/// Because the command buffers we have after the creation of the new resources still reference
	/// the old ones, we can't use them, and so we have to make sure we discard them properly.
	///
	pub fn invalidate_command_buffers(
		&mut self,
		context: &DeviceContext
	) -> Result<(), Error> {
		self.progressive_422_command_buffer = None;
		unsafe {
			context.device().reset_command_pool(
				self.command_pool,
				vk::CommandPoolResetFlags::RELEASE_RESOURCES
			)
		}?;
		self.leaked_command_buffers = 0;
		Ok(())
	}

	/// On some occasions a command buffer may be leaked. This function helps us keep track of that
	/// and forces the command buffers to be invalidated if we end up leaking too many in between
	/// calls to [`Self::invalidate_command_buffers`].
	fn signal_leaked_command_buffer(
		&mut self,
		context: &DeviceContext,
	) -> Result<(), Error> {
		self.leaked_command_buffers += 1;
		if self.leaked_command_buffers > Self::MAX_LEAKED_COMMAND_BUFFERS {
			self.invalidate_command_buffers(context)
		} else {
			Ok(())
		}
	}

	/// Acquires a previously recorded buffer or allocates a new one if not available.
	pub unsafe fn command_buffer(
		&mut self,
		context: &DeviceContext,
		shared: &SharedState,
		frame: &Frame<<ProRes as Decoder>::FrameState>,
		frame_layout: FrameLayout,
		scan_behavior: ScanBehavior,
		subsampling_format: SubsamplingFormat
	) -> Result<vk::CommandBuffer, Error> {
		let command_pool = self.command_pool;
		let record_new_buffer = || unsafe {
			let buffer = context.device().allocate_command_buffers(
				&vk::CommandBufferAllocateInfo::builder()
					.command_pool(command_pool)
					.level(vk::CommandBufferLevel::PRIMARY)
					.command_buffer_count(1)
					.build())?[0];

			context.device().begin_command_buffer(
				buffer,
				&vk::CommandBufferBeginInfo::builder()
					.build())?;

			self.record_commands(
				context,
				shared,
				frame,
				buffer,
				scan_behavior,
				subsampling_format,
				frame_layout)?;

			context.device().end_command_buffer(buffer)?;

			buffer
		};
		macro_rules! update_command_buffer {
			($target:expr) => {{
				if $target.is_some() { self.signal_leaked_command_buffer(context) }
				$target.replace((frame_layout, record_new_buffer()))
			}}
		}

		match (scan_behavior, subsampling_format) {
			(ScanBehavior::Progressive, SubsamplingFormat::Subsampling422) => match &mut self.progressive_422_command_buffer {
				Some((tentative_frame_layout, buffer))
					if tentative_frame_layout == frame_layout => Ok(*buffer),
				 old => update_command_buffer!(old)
			},
			_ => unimplemented!()
		}
	}

	/// Records the commands for a given configuration on a given command buffer. The buffer must be
	/// put in recording mode before it is passed to this function.
	unsafe fn record_commands(
		&self,
		context: &DeviceContext,
		shared: &SharedState,
		frame: &Frame<<ProRes as Decoder>::FrameState>,
		buffer: vk::CommandBuffer,
		scan: ScanBehavior,
		subsampling: SubsamplingFormat,
		frame_layout: FrameLayout,
	) -> Result<(), Error> {
		if subsampling != SubsamplingFormat::Subsampling422 || scan != ScanBehavior::Progressive {
			unimplemented!()
		}

		let FrameLayout { mb_width, mb_height, slice_count_per_row: slices_per_mb_row } = frame_layout;
		let slice_count = u32::from(slices_per_mb_row) * u32::from(mb_height);

		let on_device_memory = self.on_device_memory.as_ref().unwrap();
		let upload_memory = self.upload_memory.as_ref().unwrap();

		let dev = context.device();
		dev.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::COMPUTE, shared.unpack_pipeline);
		dev.cmd_bind_descriptor_sets(
			buffer,
			vk::PipelineBindPoint::COMPUTE,
			shared.unpack_pipeline_layout,
			0,
			&[
				upload_memory.descriptor_set,
				on_device_memory.descriptor_set
			],
			&[0, 0]);
		dev.cmd_pipeline_barrier(
			buffer,
			vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::TOP_OF_PIPE,
			vk::PipelineStageFlags::COMPUTE_SHADER,
			vk::DependencyFlags::empty(),
			&[],
			&[
				/* When we're uploading frame data to the device, the code is laid out such that our
				 * last call to vkFlushMappedMemoryRanges happens before we submit this command
				 * buffer is submitted via vkQueueSubmit. Because of that, all of those writes have
				 * already been made visible to all agents and references on the device[1] by the
				 * time this command buffer starts executing, and so there's no need to explicitly
				 * include the memory dependency on host writes here.
				 *
				 * So we don't have to put any barriers here.
				 *
				 * [1]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#memory-model-vulkan-availability-visibility */
			],
			&[
				vk::ImageMemoryBarrier::builder()
					.dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
					.image(on_device_memory.coefficient_image)
					.old_layout(vk::ImageLayout::UNDEFINED)
					.new_layout(vk::ImageLayout::GENERAL)
					.subresource_range(
						vk::ImageSubresourceRange::builder()
							.aspect_mask(vk::ImageAspectFlags::COLOR)
							.base_array_layer(0)
							.base_mip_level(0)
							.layer_count(1)
							.level_count(1)
							.build())
					.build(),
				vk::ImageMemoryBarrier::builder()
					.dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
					.image(on_device_memory.component_image)
					.old_layout(vk::ImageLayout::UNDEFINED)
					.new_layout(vk::ImageLayout::GENERAL)
					.subresource_range(
						vk::ImageSubresourceRange::builder()
							.aspect_mask(vk::ImageAspectFlags::COLOR)
							.base_array_layer(0)
							.base_mip_level(0)
							.layer_count(1)
							.level_count(1)
							.build())
					.build(),
			]);

		dev.cmd_dispatch(buffer, slice_count, 1, 1);
		dev.cmd_pipeline_barrier(
			buffer,
			vk::PipelineStageFlags::COMPUTE_SHADER,
			vk::PipelineStageFlags::COMPUTE_SHADER,
			vk::DependencyFlags::empty(),
			&[
				vk::MemoryBarrier::builder()
					.src_access_mask(vk::AccessFlags::SHADER_WRITE)
					.dst_access_mask(vk::AccessFlags::SHADER_READ)
					.build()
			],
			&[],
			&[]);

		dev.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::COMPUTE, shared.idct_pipeline);
		dev.cmd_bind_descriptor_sets(
			buffer,
			vk::PipelineBindPoint::COMPUTE,
			shared.idct_pipeline_layout,
			0,
			&[
				on_device_memory.descriptor_set
			],
			&[0]);
		dev.cmd_dispatch(
			buffer,
			u32::from(mb_width) * 16,
			u32::from(mb_height) * 16,
			1
		);
		dev.cmd_pipeline_barrier(
			buffer,
			vk::PipelineStageFlags::COMPUTE_SHADER,
			vk::PipelineStageFlags::COMPUTE_SHADER,
			vk::DependencyFlags::empty(),
			&[
				vk::MemoryBarrier::builder()
					.src_access_mask(vk::AccessFlags::SHADER_WRITE)
					.dst_access_mask(vk::AccessFlags::SHADER_READ)
					.build()
			],
			&[],
			&[]);

		dev.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::COMPUTE, shared.idct_pipeline);
		dev.cmd_bind_descriptor_sets(
			buffer,
			vk::PipelineBindPoint::COMPUTE,
			shared.idct_pipeline_layout,
			0,
			&[
				on_device_memory.descriptor_set
			],
			&[0]);
		dev.cmd_dispatch(
			buffer,
			u32::from(mb_width) * 16,
			u32::from(mb_height) * 16,
			1
		);

		dev.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::COMPUTE, shared.spread_422_pipeline);
		dev.cmd_bind_descriptor_sets(
			buffer,
			vk::PipelineBindPoint::COMPUTE,
			shared.spread_422_pipeline_layout,
			0,
			&[
				on_device_memory.descriptor_set,
				frame.descriptor_set(),
			],
			&[0, 0]);
		dev.cmd_dispatch(
			buffer,
			u32::from(mb_width) * 8,
			u32::from(mb_height) * 16,
			1
		);

		Ok(())
	}

	/// Pick the memory type with the largest heap that fits the given constraints.
	fn pick_memory(
		shared: &SharedState,
		mask: u32,
		type_requirements: vk::MemoryPropertyFlags,
		heap_requirements: vk::MemoryHeapFlags,
	) -> Option<u32> {
		let mut candidates = 0u32;
		for i in 0..shared.memory_properties.memory_type_count {
			let memory_type = shared.memory_properties.memory_types[i as usize];
			let memory_heap = shared.memory_properties.memory_heaps[memory_type.heap_index as usize];

			let bit = 1u32 << i;
			if bit & mask == 0 { continue }

			let a = memory_type.property_flags.contains(type_requirements);
			let b = memory_heap.flags.contains(heap_requirements);
			if !a || !b { continue }

			candidates |= bit;
		}

		let mut chosen_type = None;
		for i in 0..u32::BITS {
			let is_candidate = (candidates >> i) & 1 != 0;
			if !is_candidate { continue }

			let memory_type = shared.memory_properties
				.memory_types[i as usize];
			let memory_heap = shared.memory_properties
				.memory_heaps[memory_type.heap_index as usize];

			if let Some(chosen_type) = chosen_type {
				let chosen_memory_type = shared.memory_properties
					.memory_types[chosen_type as usize];
				let chosen_memory_heap = shared.memory_properties
					.memory_heaps[chosen_memory_type.heap_index as usize];

				if chosen_memory_heap.size >= memory_heap.size {
					continue
				}
			}

			chosen_type = Some(i)
		}

		chosen_type
	}

	/// Reports whether the current images are large enough for the given data.
	pub fn on_device_memory_large_enough(
		&self,
		coefficient_image_extent: vk::Extent2D,
		component_image_extent: vk::Extent2D,
	) -> bool {
		self.on_device_memory.as_ref()
			.map(|on_device_memory|
				   on_device_memory.coefficient_image_extent.width >= coefficient_image_extent.width
				&& on_device_memory.coefficient_image_extent.height >= coefficient_image_extent.height
				&& on_device_memory.component_image_extent.width >= component_image_extent.width
				&& on_device_memory.component_image_extent.height >= component_image_extent.height)
			.unwrap_or(true)
	}

	/// Reallocates the on device memory allocation and sets up new buffers for it.
	pub unsafe fn realloc_on_device_memory(
		&mut self,
		context: &DeviceContext,
		shared: &SharedState,
		coefficient_image_extent: vk::Extent2D,
		component_image_extent: vk::Extent2D,
	) {
		let props = context.instance().get_physical_device_image_format_properties(
			*context.physical_device(),
			vk::Format::R32G32B32A32_SFLOAT,
			vk::ImageType::TYPE_2D,
			vk::ImageTiling::OPTIMAL,
			vk::ImageUsageFlags::STORAGE,
			vk::ImageCreateFlags::default())
			.unwrap();

		let extent_valid = |extent|
			   props.max_extent.width >= extent.width
			&& props.max_extent.height >= extent.height
			&& props.max_extent.depth >= 1;
		if !extent_valid(coefficient_image_extent) || !extent_valid(component_image_extent) {
			panic!("extent too large")
		}

		let build_image = |size| context.device()
			.create_image(&vk::ImageCreateInfo::builder()
				.usage(vk::ImageUsageFlags::STORAGE)
				.image_type(vk::ImageType::TYPE_2D)
				.format(vk::Format::R32G32B32A32_SFLOAT)
				.mip_levels(1)
				.extent(vk::Extent3D {
					width: size.width,
					height: size.height,
					depth: 1,
				})
				.build(), None)
			.unwrap();

		let coefficient_image = build_image(coefficient_image_extent);
		let component_image = build_image(component_image_extent);

		let coefficient_image_memreq = context.device().get_image_memory_requirements(coefficient_image);
		let component_image_memreq = context.device().get_image_memory_requirements(component_image);
		let padding0 =
			(component_image_memreq.alignment - coefficient_image_memreq.size % component_image_memreq.alignment)
				% component_image_memreq.alignment;

		let total_size =
			coefficient_image_memreq.size
				.checked_add(component_image_memreq.size)
				.unwrap()
				.checked_add(padding0)
				.unwrap();

		let requires_new_allocation = self.on_device_memory
			.as_ref()
			.map(|on_device_memory| on_device_memory.allocation_size < total_size)
			.unwrap_or(true);

		let (allocation, allocation_size) = if requires_new_allocation {
			let memory_type_index = Self::pick_memory(
				shared,
				coefficient_image_memreq.memory_type_bits & component_image_memreq.memory_type_bits,
				vk::MemoryPropertyFlags::DEVICE_LOCAL,
				vk::MemoryHeapFlags::DEVICE_LOCAL)
				.unwrap();

			(context.allocate_memory(&vk::MemoryAllocateInfo::builder()
				.allocation_size(total_size)
				.memory_type_index(memory_type_index)
				.build())
				.unwrap()
				.unwrap(), total_size)
		} else {
			(
				self.on_device_memory.as_ref().unwrap().allocation,
				self.on_device_memory.as_ref().unwrap().allocation_size
			)
		};

		if let Some(on_device_memory) = self.on_device_memory.take() {
			context.device().destroy_image(on_device_memory.coefficient_image, None);
			context.device().destroy_image(on_device_memory.component_image, None);

			if requires_new_allocation {
				context.free_memory(on_device_memory.allocation);
			}
		}

		context.device().bind_image_memory(
			coefficient_image,
			allocation,
			0
		).unwrap();

		context.device().bind_image_memory(
			component_image,
			allocation,
			coefficient_image_memreq.size + padding0
		).unwrap();

		self.on_device_memory = Some(OnDeviceMemory {
			coefficient_image_extent,
			coefficient_image,
			component_image_extent,
			component_image,
			allocation_size,
			allocation,
		});
	}

	/// Reports whether the current buffers are large enough for the given data.
	pub fn upload_memory_large_enough(
		&self,
		index_size: vk::DeviceSize,
		picture_data_size: vk::DeviceSize
	) -> bool {
		let index_buffer_large_enough = self.upload_memory
			.map(|a| a.index_buffer_size >= index_size)
			.unwrap_or(false);
		let picture_data_buffer_large_enough = self.upload_memory
			.map(|a| a.picture_data_buffer_size >= picture_data_size)
			.unwrap_or(false);

		index_buffer_large_enough && picture_data_buffer_large_enough
	}

	/// Reallocates the upload memory allocation and sets up new buffers for it.
	pub unsafe fn realloc_upload_memory(
		&mut self,
		context: &DeviceContext,
		shared: &SharedState,
		index_size: vk::DeviceSize,
		picture_data_size: vk::DeviceSize,
	) {
		let build_buffer = |size| context.device()
			.create_buffer(&vk::BufferCreateInfo::builder()
				.size(index_size)
				.usage(vk::BufferUsageFlags::STORAGE_BUFFER)
				.build(), None)
			.unwrap();

		let index_buffer = build_buffer(index_size);
		let picture_data_buffer = build_buffer(picture_data_size);

		let index_memreq = context.device().get_buffer_memory_requirements(index_buffer);
		let picture_data_memreq = context.device().get_buffer_memory_requirements(picture_data_buffer);
		let padding0 =
			(picture_data_memreq.alignment - index_memreq.size % picture_data_memreq.alignment)
				% picture_data_memreq.alignment;

		let total_size =
			index_memreq.size
				.checked_add(picture_data_memreq.size)
				.unwrap()
				.checked_add(padding0)
				.unwrap();

		let requires_new_allocation = self.upload_memory
			.as_ref()
			.map(|upload_memory| upload_memory.allocation_size < total_size)
			.unwrap_or(true);

		let (allocation, allocation_size) = if requires_new_allocation {
			let memory_type_index = Self::pick_memory(
				shared,
				index_memreq.memory_type_bits & picture_data_memreq.memory_type_bits,
				vk::MemoryPropertyFlags::HOST_VISIBLE,
				vk::MemoryHeapFlags::default())
				.unwrap();

			(context.allocate_memory(&vk::MemoryAllocateInfo::builder()
				.allocation_size(total_size)
				.memory_type_index(memory_type_index)
				.build())
				.unwrap()
				.unwrap(), total_size)
		} else {
			(
				self.upload_memory.as_ref().unwrap().allocation,
				self.upload_memory.as_ref().unwrap().allocation_size,
			)
		};

		let map_new_allocation = ||
			NonNull::new(context.device().map_memory(
				allocation,
				0,
				total_size,
				vk::MemoryMapFlags::empty())
				.unwrap() as *mut u8)
				.unwrap();

		let map = match self.upload_memory.take() {
			Some(upload_memory) => {
				context.device().destroy_buffer(upload_memory.index_buffer, None);
				context.device().destroy_buffer(upload_memory.picture_data_buffer, None);

				if requires_new_allocation {
					context.free_memory(upload_memory.allocation);
					map_new_allocation()
				} else { upload_memory.map }
			},
			None => map_new_allocation()
		};

		context.device().bind_buffer_memory(
			index_buffer,
			allocation,
			0
		).unwrap();

		context.device().bind_buffer_memory(
			picture_data_buffer,
			allocation,
			index_memreq.size + padding0
		).unwrap();

		self.upload_memory = Some(UploadMemory {
			index_buffer_offset: 0,
			index_buffer_size: index_memreq.size,
			index_buffer,
			picture_data_buffer_offset: index_memreq.size + padding0,
			picture_data_buffer_size: picture_data_memreq.size,
			picture_data_buffer,
			allocation_size,
			allocation,
			map,
		});
	}

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
					.command_pool(self.command_pool)
					.command_buffer_count(1)
					.build())?;
			Ok(buffers[0])
		}
	}
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

	#[sshdr::include(file = "../../../shaders/prores/Spread422.glsl", stage = "compute")]
	pub static SPREAD_422: &'static [u32];
}
