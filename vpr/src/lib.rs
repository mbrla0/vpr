//! Vulkan ProRes
//!
//! This crate provides an interface to
#![feature(const_ptr_read)]

mod error;
mod prores;
mod decoder;
mod image;
mod context;

pub use context::*;

pub use prores::*;
pub use decoder::*;
pub use error::*;

use std::cell::RefCell;
use std::sync::Arc;
use ash::vk;
use c_str_macro::c_str;

///
pub struct Instance(Arc<VprContext>);
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

		Ok(Self(Arc::new(VprContext {
			entry,
			instance,
			device: VprDeviceContext {
				shared_decoder_state: Default::default(),
				graphics_context: Arc::new(DeviceContext {
					physical_device,
					device
				})
			}
		})))
	}

	pub fn decoder<C>(&self, decoder: C) -> Result<DecodeQueue<C>, Error>
		where C: Decoder {

		todo!()
	}
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
