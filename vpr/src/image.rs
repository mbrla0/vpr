use ash::vk;

pub struct Frame<'a, T> {
	state: &'a T,
	image: Image,
}
impl<'a, T> Frame<'a, T> {
	pub fn state(&self) -> &T {
		self.state
	}
	pub fn image(&self) -> &Image {
		&self.image
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