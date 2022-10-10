use std::collections::HashMap;
use std::ffi::CStr;
use std::time::Duration;
use ash::vk;

fn main() {
	unsafe { init() }
}

unsafe fn run(
	instance: &ash::Instance,
	physical_device: vk::PhysicalDevice,
	features: vk::PhysicalDeviceFeatures,
	device: &ash::Device) -> (HashMap<&'static str, Duration>) {

	

}

unsafe fn init() {
	let entry = Box::new(ash::Entry::load().unwrap());
	let instance = Box::new({
		let ai = vk::ApplicationInfo::builder()
			.api_version(vk::API_VERSION_1_0)
			.build();
		let ici = vk::InstanceCreateInfo::builder()
			.application_info(&ai)
			.build();

		entry.create_instance(&ici, None).unwrap()
	});

	let physical_devices = instance.enumerate_physical_devices().unwrap();
	let mut results = Vec::new();
	for physical_device in physical_devices {
		let handle = std::thread::spawn(|| {
			let queue_family = instance
				.get_physical_device_queue_family_properties(physical_device)
				.into_iter()
				.enumerate()
				.find(|(_, queue)| queue.queue_flags.contains(vk::QueueFlags::COMPUTE))
				.expect("device does not support compute")
				.0;
			let features = instance.get_physical_device_features(physical_device);


			let dci = vk::DeviceCreateInfo::builder()
				.queue_create_infos(&[
					vk::DeviceQueueCreateInfo::builder()
						.queue_family_index(queue_family as u32)
						.queue_priorities(&[1.0])
						.build()
				])
				.enabled_features(&features)
				.build();

			let device = Box::new(instance.create_device(
				physical_device,
				&dci,
				None).unwrap());

			run(&instance, physical_device, features, &device)
		});

		let label = instance.get_physical_device_properties(physical_device);
		let label = CStr::from_ptr(label.device_name.as_ptr())
			.to_str()
			.unwrap()
			.to_owned();

		results.push((label, handle));
	}

	for (label, result) in results {
		println!("{}: ", label);

		let result = match result.join() {
			Ok(result) => result,
			Err(what) => {
				if what.is::<String>() {
					let what = what.downcast_ref_unchecked::<String>();
					println!("    {}", what);
				} else if what.is::<str>() {
					let what = what.downcast_ref_unchecked::<str>();
					println!("    {}", what);
				}

				continue
			}
		};

		for (name, time) in result {
			println!("    {}: {:?}", name, time);
		}
	}
}
