
pub struct Error;
impl Error {
	pub(crate) fn from_vk_general(what: ash::vk::Result) -> Self {
		Self
	}
}
impl From<ash::LoadingError> for Error {
	fn from(_: ash::LoadingError) -> Self {
		Self
	}
}