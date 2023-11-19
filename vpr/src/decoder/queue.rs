use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use ash::vk;
use crate::context::VprContext;
use crate::{Decoder, DeviceContext};
use crate::util::pick_largest_memory_heap;

/// The state related to a single worker in the decoding process.
struct DecodeWorker<C>
	where C: Decoder {

	state: C::WorkerState,
	handle: JoinHandle<Result<(), C::Error>>
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct FrameKey {
	extent: vk::Extent2D,
}

struct FrameInstance {
	key: FrameKey,
	image: vk::Image,
	image_view: vk::ImageView,
	descriptor_set: vk::DescriptorSet,
}

struct FrameBucket {
	allocation: vk::DeviceMemory,
	allocation_size: vk::DeviceSize,

	capacity: vk::DeviceSize,
	frames: vk::DeviceSize,

	element_create_info: vk::ImageCreateInfo,
	element_memreq: vk::MemoryRequirements,
	element_size: vk::DeviceSize,

	in_flight: vk::DeviceSize,
	in_wait: VecDeque<FrameInstance>,
}
impl FrameBucket {
	const MIN_CAPACITY: vk::DeviceSize = 8;

	pub unsafe fn new(context: &DeviceContext, format: vk::Format, extent: vk::Extent2D) -> Self {
		let element_create_info = vk::ImageCreateInfo::builder()
			.format(format)
			.flags(vk::ImageCreateFlags::empty())
			.extent(vk::Extent3D {
				width: extent.width,
				height: extent.height,
				depth: 1,
			})
			.mip_levels(1)
			.array_layers(1)
			.usage(vk::ImageUsageFlags::STORAGE)
			.image_type(vk::ImageType::TYPE_2D)
			.tiling(vk::ImageTiling::OPTIMAL)
			.build();

		let image = context.device().create_image(&element_create_info, None)?;
		let element_memreq = context.device().get_image_memory_requirements(image);
		context.device().destroy_image(image, None);

		let padding = (element_memreq.alignment - element_memreq.size % element_memreq.alignment)
			% element_memreq.alignment;
		let element_size = element_memreq.size + padding;

		let capacity = Self::MIN_CAPACITY;

		let allocation_size = element_size * capacity;
		let allocation = context.device().allocate_memory(
			&vk::MemoryAllocateInfo::builder()
				.allocation_size(allocation_size)
				.memory_type_index(pick_largest_memory_heap(
					context.physical_device_memory_properties(),
					element_memreq.memory_type_bits,
					vk::MemoryPropertyFlags::DEVICE_LOCAL,
					vk::MemoryHeapFlags::DEVICE_LOCAL,
				).unwrap())
				.build(),
			None)?;

		Ok(Self {
			allocation,
			allocation_size,
			capacity,
			frames: 0,
			element_create_info,
			element_memreq,
			element_size,
			in_flight: 0,
			in_wait: Default::default(),
		})
	}
}



/// The structure responsible for allocating and managing the lifetimes of the frames that are
/// handed out for the decoder to write to and for our clients to read from.
///
/// # Allocation Strategy
/// The manager uses a fairly rudimentary allocation strategy based on the observation that most
/// video streams will only use a very small number of distinct target resolutions throughout their
/// lifetimes, and most often they will only use one.
///
/// Therefore we use one allocation bucket for every kind of frame that we want to use.
struct FrameAllocator {
	/// All of the frame keys that are currently alive.
	keys: Vec<FrameKey>,
	/// The buckets where we store the data for the frames.
	buckets: Vec<FrameBucket>,
}
impl FrameAllocator {
	pub fn acquire(&mut self, extent: vk::Extent2D) -> FrameGuard {
		let key = FrameKey { extent };
		let index = self.keys
			.iter()
			.position(|candidate| *candidate == key)
			.unwrap_or(self.keys.len());
		if index == self.keys.len() {
			/* Create a new bucket for this frame and others like it. */

		}

		FrameGuard {
			allocator: self,
			instance: FrameInstance {},
			key,
		}
	}
}

struct FrameGuard<'a> {
	allocator: &'a FrameAllocator,
	instance: FrameInstance,
	key: FrameKey,
}
impl Drop for FrameGuard<'_> {
	fn drop(&mut self) {
		self.allocator.
	}
}

/// The decode queue.
///
/// This is the main structure through which decoding is actually performed and
/// decoding operations are set up.
pub struct DecodeQueue<C>
	where C: Decoder {

	context: Arc<VprContext>,
	decoder: C,

	decoder_instance_state: C::InstanceState,

	frame_allocator: FrameAllocator,

	incoming_data_buffer: Vec<u8>,
	workers: Vec<DecodeWorker<C>>
}
impl<C> DecodeQueue<C> {
	pub(crate) fn new(
		context: Arc<VprContext>,
		decoder: C) -> Result<Self, Error> {

		Ok(Self {
			context,
			decoder,
			incoming_data_buffer: vec![]
		})
	}
}