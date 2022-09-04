use std::sync::Arc;
use crate::{Context, Error};

pub trait Decoder {
	/// Error type for this decoder.
	type Error;
	/// State data shared between all grouped instances of a decoder type.
	type SharedState;
	/// State data local to one decoder instance of a decoder type.
	type InstanceState;

	fn begin(&self,
		context: &Context,
		shared: &Self::SharedState,
		instance: &mut Self::InstanceState);

	fn end(&self,
		context: &Context,
		shared: &Self::SharedState,
		instance: &mut Self::InstanceState);

	fn create_shared_state(context: &Context) -> Result<Self::SharedState, Self::Error>;
	fn create_instance_state(context: &Context) -> Result<Self::InstanceState, Self::Error>;

	fn destroy_shared_state(context: &Context, shared: &mut Self::SharedState);
	fn destroy_instance_state(context: &Context, instance: &mut Self::InstanceState);
}

pub struct DecodeQueue<C> {
	context: Arc<Context>,
	decoder: C,
}
impl<C> DecodeQueue<C> {
	pub(crate) fn new(
		context: Arc<Context>,
		decoder: C) -> Result<Self, Error> {


	}
}
