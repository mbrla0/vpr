use ash::vk;

pub struct Frame<'a, T> {
	state: &'a T,
	image: ImageView,
	completion_fence: vk::Fence,
	completion_semaphore: vk::Semaphore,
}
impl<'a, T> Frame<'a, T> {
	/// The state associated with this frame.
	pub fn state(&self) -> &T {
		self.state
	}

	/// The image backing this frame.
	pub fn image(&self) -> &ImageView {
		&self.image
	}
	
	/// Obtain the fence that must be signaled by the decoder to indicate the
	/// frame is done.
	pub fn completion_fence(&self) -> vk::Fence {
		self.completion_fence
	}

	/// Obtain the semaphore that must be signaled by the decoder to indicate
	/// the frame is done.
	pub fn completion_semaphore(&self) -> vk::Semaphore {
		self.completion_semaphore
	}
}

/// Structure describing an image view in device memory.
pub struct ImageView {
	image: vk::ImageView,
	format: vk::Format,
	size: (u32, u32)
}
impl ImageView {
	pub fn raw_handle(&self) -> vk::ImageView {
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