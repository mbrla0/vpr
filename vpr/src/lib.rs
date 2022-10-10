//! Vulkan ProRes
//!
//! This crate provides an interface to
#![feature(const_ptr_read)]

mod error;
mod prores;
mod decode;
mod image;
pub use prores::*;
pub use decode::*;
pub use error::*;

use std::cell::RefCell;
use std::sync::Arc;
use ash::vk;
use c_str_macro::c_str;

///
pub struct Instance(Arc<Context>);
impl Instance {
	pub fn new<F>(mut devices: F) -> Result<Self, Error>
		where F: FnMut(&[SuitablePhysicalDevice]) {

		let entry = unsafe { ash::Entry::load() }
			.map_err(|_| Error)?;

		let instance = unsafe {
			let ai = vk::ApplicationInfo::builder()
				.api_version(vk::API_VERSION_1_0)
				.application_name(c_str!("vpr"))
				.application_version(100)
				.build();
			let ici = vk::InstanceCreateInfo::builder()
				.application_info(&ai)
				.build();

			entry.create_instance(&ici, None)
				.map_err(Error::from_vk_general)?
		};

		let enable = RefCell::new(None);
		let candidates = unsafe { instance.enumerate_physical_devices() }
			.map_err(Error::from_vk_general)?
			.into_iter()
			.filter_map(|device| {
				let queue_family_properties = unsafe {
					instance
						.get_physical_device_queue_family_properties(device)
				};
				let queue_family_index = queue_family_properties
					.into_iter()
					.enumerate()
					.filter(|(_, queue)| queue.queue_flags.contains(
						vk::QueueFlags::COMPUTE
							| vk::QueueFlags::TRANSFER))
					.next()?.0;

				Some(SuitablePhysicalDevice {
					enable: &enable,
					device,
					info: PhysicalDeviceSuitabilityInformation {
						queue_family_index: queue_family_index as u32
					}
				})
			})
			.collect::<Vec<_>>();
		(devices)(&candidates[..]);

		let (physical_device, suitability_info) = enable.into_inner().ok_or(Error)?;
		let device = unsafe {
			let dci = vk::DeviceCreateInfo::builder()
				.queue_create_infos(&[
					vk::DeviceQueueCreateInfo::builder()
						.queue_family_index(suitability_info.queue_family_index)
						.queue_priorities(&[1.0])
						.build()
				])
				.build();
			instance.create_device(
				physical_device,
				&dci,
				None)
				.map_err(Error::from_vk_general)?
		};

		Ok(Self(Arc::new(Context {
			vulkan: VulkanContext {
				entry,
				instance,
				physical_device,
				device
			}
		})))
	}

	pub fn decoder<C>(&self, decoder: C) -> Result<DecodeQueue<C>, Error>
		where C: Decoder {

		todo!()
	}
}

/// The shared context any given instance is reliant on.
struct Context {
	vulkan: VulkanContext
}

/// The shared context any given instance is reliant on.
pub struct VulkanContext {
	entry: ash::Entry,
	instance: ash::Instance,
	physical_device: vk::PhysicalDevice,
	device: ash::Device,
}
impl VulkanContext {
	pub fn instance(&self) -> &ash::Instance { &self.instance }
	pub fn physical_device(&self) -> &vk::PhysicalDevice { &self.physical_device }
	pub fn device(&self) -> &ash::Device { &self.device }
}

/// The requirements for a given physical device to be suitable.
#[derive(Debug, Copy, Clone)]
struct PhysicalDeviceSuitabilityInformation {
	queue_family_index: u32
}

///
pub struct SuitablePhysicalDevice<'a> {
	enable: &'a RefCell<Option<(vk::PhysicalDevice, PhysicalDeviceSuitabilityInformation)>>,
	device: vk::PhysicalDevice,
	info: PhysicalDeviceSuitabilityInformation
}
impl<'a> SuitablePhysicalDevice<'a> {
	/// Selects the this physical device for use in the instance being set up.
	pub fn select(&self) {
		self.enable
			.borrow_mut()
			.replace((self.device, self.info));
	}
}
