use ash::vk;

pub struct Frame<'a, T> {
	state: &'a T,
	image: Image,
	completion_fence: (bool, vk::Fence),
	completion_semaphore: (bool, vk::Fence),
}
impl<'a, T> Frame<'a, T> {
	/// The state associated with this frame.
	pub fn state(&self) -> &T {
		self.state
	}

	/// The image backing this frame.
	pub fn image(&self) -> &Image {
		&self.image
	}

	/// Obtain the fence that must be signaled by the decoder to indicate the
	/// frame is done.
	pub unsafe fn completion_fence(&mut self) -> vk::Fence {
		self.completion_fence.0 = true;
		self.completion_fence.1
	}

	/// Obtain the semaphore that must be signaled by the decoder to indicate
	/// the frame is done.
	pub unsafe fn completion_semaphore(&mut self) -> vk::Semaphore {
		self.completion_semaphore.0 = true;
		self.completion_semaphore.1
	}
}

/// Structure describing an image in device memory.
pub struct Image {
	image: vk::Image,
	format: vk::Format,
	size: (u32, u32)
}
impl Image {
	pub fn raw_handle(&self) -> vk::Image {
		self.image
	}
	pub fn format(&self) -> vk::Format {
		self.format
	}
	pub fn width(&self) -> u32 {
		self.size.0
	}
	pub fn height(&self) -> u32 {
		self.size.1
	}
}