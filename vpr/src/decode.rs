use std::sync::Arc;
use crate::{VulkanContext, Error, Context};

pub trait Decoder {
	/// State data shared between all grouped instances of a decoder type.
	type SharedState;
	/// State data local to one decoder instance of a decoder type.
	type InstanceState;

	fn create_shared_state(context: &VulkanContext) -> Result<Self::SharedState, ()>;
	fn create_instance_state(context: &VulkanContext) -> Result<Self::InstanceState, ()>;

	fn destroy_shared_state(context: &VulkanContext, shared: &mut Self::SharedState);
	fn destroy_instance_state(context: &VulkanContext, instance: &mut Self::InstanceState);

	fn begin(&self,
		context: &VulkanContext,
		shared: &Self::SharedState,
		instance: &mut Self::InstanceState);

	fn end(&self,
		context: &VulkanContext,
		shared: &Self::SharedState,
		instance: &mut Self::InstanceState);
}

pub struct DecodeQueue<C> {
	context: Arc<Context>,
	decoder: C,
}
impl<C> DecodeQueue<C> {
	pub(crate) fn new(
		context: Arc<Context>,
		decoder: C) -> Result<Self, Error> {

		todo!()
	}
}
