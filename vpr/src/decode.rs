use std::sync::Arc;

use crate::{Context, Error};
use crate::image::{Frame, Image};

/// Decoder.
///
/// # Execution Model
/// This trait is structured so as to conform to an execution model whose
/// primary goal is to achieve a high level of throughput through thorough usage
/// of all processor and graphics device resources available on a system. It
/// does this by encouraging strict separation of data structures along lines of
/// locality to a given processing node, taking into consideration how they map
/// into Vulkan concepts, in order to keep any eventual overhead as a result of
/// translating between them down to a minimum.
///
/// The before-mentioned of locality are as follows:
///
/// - Shared: Shared amongst all invocations on a device and amongst all cores
///           in the system.
/// - Instance: aaaaaa
///
pub trait Decoder {
	/// State data shared between all grouped instances of a decoder type.
	type SharedState;
	/// State data local to one decoder instance of a decoder type.
	type InstanceState;
	/// State date associated with each frame of the decoder.
	type FrameState;
	/// Error type used by this decoder.
	type Error;

	/// Consume raw encoded data and schedule frames for decoding
	fn schedule(&self,
		context: &Context,
		shared: &Self::SharedState,
		instance: &mut Self::InstanceState,
		frames: &DecodeScheduler<Self>,
		data: &[u8]);

	/// Perform the decoding of the frame.
	fn decode(&self,
		context: &Context,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		frame: &Frame<Self::FrameState>,
		data: &[u8]);

	fn create_shared_state(
		context: &Context) -> Result<Self::SharedState, Self::Error>;
	fn create_instance_state(
		context: &Context,
		shared: &Self::SharedState) -> Result<Self::InstanceState, Self::Error>;
	fn create_frame_state(
		context: &Context,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		image: &Image) -> Result<Self::FrameState, Self::Error>;

	fn destroy_shared_state(
		context: &Context,
		shared: &mut Self::SharedState);
	fn destroy_instance_state(
		context: &Context,
		shared: &Self::SharedState,
		instance: &mut Self::InstanceState);
	fn destroy_frame_state(
		context: &Context,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		frame: &mut Self::FrameState);
}

/// The scheduler for image operations.
///
pub struct DecodeScheduler<C> {
	_bind: std::marker::PhantomData<C>,
}
impl<C> DecodeScheduler<C>
	where C: Decoder {

	pub fn schedule<T>(&mut self, bundle: T) {

	}
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
