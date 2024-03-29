use std::alloc::Layout;
use std::any::TypeId;
use std::collections::HashMap;
use std::mem::{ManuallyDrop};
use std::sync::{Arc, RwLock};
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::SeqCst;
use ash::prelude::VkResult;
use ash::vk;
use crate::Decoder;

/// The internal shared context any given instance is reliant on.
pub(crate) struct VprContext {
	pub entry: ash::Entry,
	pub device: VprDeviceContext
}

/// The internal context structures pertaining to a single device instance.
pub(crate) struct VprDeviceContext {
	/// Storage for the state shared by the all decoders that have at some point
	/// been instanced to run on this device.
	pub shared_decoder_state: SharedHeterogeneousStore,

	/// The Vulkan data structures instanced for this device.
	pub graphics_context: Arc<DeviceContext>,
}
impl VprDeviceContext {
	/// Makes sure the shared state for a given decoder is available.
	///
	/// On the first call to this function for a given decoder, it will call the
	/// provided initialization procedure and store it for future use.
	pub fn ensure_shared_decoder_state<D, F>(&self, init: F)
		where D: Decoder,
			  F: FnOnce() -> D::SharedState {

		self.shared_decoder_state.initialize_with(|| {
			let state = (init)();

			let graphics_context = self.graphics_context.clone();
			let destructor = Box::new(move |data: *mut u8| unsafe {
				let data =  &mut *(data as *mut D::SharedState);
				D::destroy_shared_state(&*graphics_context, data);
			});

			(state, destructor)
		});
	}

	/// Returns the shared state for a given decoder.
	pub fn shared_decoder_state<D>(&self) -> &D::SharedState
		where D: Decoder {

		self.shared_decoder_state.get::<D::SharedState>()
			.expect("vpr requested shared state for a decoder that hasn't \
			been initialized yet. this is a bug")
	}
}

/// Structures related to the graphics context of a single device instance. This
/// is the structure where the Vulkan device instance and the physical device
/// it instances live.
///
/// It is the structure providing codecs with the ability to directly call into
/// Vulkan procedures and to manage Vulkan-related resources.
pub struct DeviceContext {
	instance: ash::Instance,
	physical_device: vk::PhysicalDevice,
	physical_device_properties: vk::PhysicalDeviceProperties,
	physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
	queue_family_properties: Vec<vk::QueueFamilyProperties>,
	device: ash::Device,

	/// The layout describing the descriptor set that can be obtained by calling
	/// [`Frame::descriptor_set`].
	///
	/// This is fixed for the lifetime of the instance and bound to it, so it makes sense to put it
	/// here. Additionally, putting it here lets decoders and encoders access this value as soon as
	/// possible.
	///
	/// [`Frame::descriptor_set`]: crate::image::Frame::descriptor_set
	frame_descriptor_set_layout: vk::DescriptorSetLayout,

	/// We use this to keep track of the allocations we've made.
	///
	/// In Vulkan, trying to allocate one past the limit is undefined behavior, and so we have to at
	/// least keep track of the number of allocations so we don't go past that limit.
	allocation_count: AtomicU32,
}
impl DeviceContext {
	/// Returns a handle to the Vulkan instance. Additionally, this handle provides access to
	/// instance methods.
	pub fn instance(&self) -> &ash::Instance {
		&self.instance
	}

	/// Returns a reference to the physical device this context is bound to.
	pub fn physical_device(&self) -> &vk::PhysicalDevice {
		&self.physical_device
	}

	/// Returns the general properties of this context's physical device.
	pub fn physical_device_properties(&self) -> &vk::PhysicalDeviceProperties {
		&self.physical_device_properties
	}

	/// Returns the memory layout of this context's physical device.
	pub fn physical_device_memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
		&self.physical_device_memory_properties
	}

	/// Returns a handle to the logical device this context is bound to. Additionally, this handle
	/// provides access to logical device methods.
	pub fn device(&self) -> &ash::Device {
		&self.device
	}

	/// Returns the list of queue families in this device.
	pub fn queue_family_properties(&self) -> &[vk::QueueFamilyProperties] {
		&self.queue_family_properties
	}

	/// Returns the descriptor set layout that describes the resources available in a [`Frame`].
	///
	/// [`Frame`]: crate::image::Frame
	pub fn frame_descriptor_set_layout(&self) -> &vk::DescriptorSetLayout {

	}

	/// Tries to allocate memory on the device, failing if the maximum number of allocations would
	/// be exceeded.
	///
	///
	pub unsafe fn allocate_memory(&self, param: &vk::MemoryAllocateInfo)
		-> Result<VkResult<vk::DeviceMemory>, TooManyAllocations> {

		let max_allocations = self.physical_device_properties.limits.max_memory_allocation_count;
		let result = self.allocation_count.fetch_update(SeqCst, SeqCst, |val| {
			if val >= max_allocations {
				None
			} else {
				Some(max_allocations + 1)
			}
		});
		match result {
			Ok(_) => Ok(self.device.allocate_memory(param, None)),
			Err(_) => Err(TooManyAllocations)
		}
	}

	/// Frees allocations previously acquired from [`Self::allocate_memory`].
	pub unsafe fn free_memory(&self, param: vk::DeviceMemory) {
		self.device.free_memory(param, None)
	}
}
impl Drop for DeviceContext {
	fn drop(&mut self) {
		unsafe {
			let _ = self.device.device_wait_idle();
			self.device.destroy_device(None);
		}
	}
}

/// This error indicates that too many allocations have been made on a target device.
///
/// It is only triggered when calling the [`DeviceContext::allocate_memory`] function, which keeps
/// track of the number of allocations made on the device, so that the user will get this error
/// instead of triggering undefined behavior.
#[derive(Debug, thiserror::Error)]
#[error("the allocation limit has been exceeded on the target device")]
pub struct TooManyAllocations;

/// Write-once-read-many storage structure for shared values of heterogeneous
/// type.
struct SharedHeterogeneousStore {
	container: RwLock<HashMap<TypeId, SharedHeterogeneousEntry>>,
}
impl SharedHeterogeneousStore {
	pub fn new() -> Self {
		Self {
			container: Default::default()
		}
	}

	/// Returns the shared value for a given type.
	pub fn get<T>(&self) -> Option<&T> {
		let id = TypeId::of::<T>();
		let (_, pointer) = *self.container.read().unwrap().get(&id)?;

		/* Safety:
		 *
		 * Because of the way we've set up our allocations, even if we mutate
		 * the map, the objects themselves won't move, only the pointers to them
		 * in the map will. This way, we can guarantee that the references we
		 * return are valid for the lifetime of the structure itself, even if
		 * we mutate the structure while holding a pointer to an object.
		 *
		 * For this assumption to work, it's also important that we never
		 * deallocate any of the objects once they've been submitted to the map.
		 */
		Some(unsafe { &*pointer })
	}

	/// Initializes an empty slot with the value returned from the given
	/// function, if needed. If the slot is not empty, the given function will
	/// not be called.
	///
	/// The initialization procedure must return both the value itself as well
	/// as boxed destruction procedure. This destruction procedure is run as
	/// this data structure is being dropped, as a way to destroy the objects
	/// that were created for this data structure.
	///
	pub fn initialize_with<T, F>(&self, f: F) -> bool
		where F: FnOnce() -> (T, Box<dyn FnOnce(*mut u8)>) {

		let id = TypeId::of::<T>();

		let mut container = self.container.write().unwrap();
		let needs_initialization = !container.contains_key(&id);
		if needs_initialization {
			let (value, destructor) = (f)();
			let allocation = Self::move_into_new_alloc(value);
			container.insert(id, SharedHeterogeneousEntry {
				layout: Layout::new::<T>(),
				allocation,
				destructor
			});
		}
		needs_initialization
	}

	/// Moves a value into a new allocation.
	fn move_into_new_alloc<T>(value: T) -> *mut u8 {
		let val = ManuallyDrop::new(value);
		unsafe {
			let layout = Layout::new::<T>();
			let allocation = std::alloc::alloc(layout);

			if allocation.is_null() {
				panic!("Could not allocate the memory required to store \
						an instance of the {} type.",
					std::any::type_name::<T>())
			}

			std::ptr::copy_nonoverlapping(
				&val as *const ManuallyDrop<T> as *const T,
				allocation as *mut T,
				1);

			allocation
		}
	}
}
impl Drop for SharedHeterogeneousStore {
	fn drop(&mut self) {
		for (_, entry) in self.container.write().unwrap().drain() {
			(entry.destructor)(entry.allocation);
			unsafe {
				std::alloc::dealloc(entry.allocation, entry.layout)
			}
		}
	}
}

/// Structure containing all of the data necessary to represent an entry in an
/// instance of the shared heterogeneous store.
struct SharedHeterogeneousEntry {
	layout: Layout,
	allocation: *mut u8,
	destructor: Box<dyn FnOnce(*mut u8)>
}

