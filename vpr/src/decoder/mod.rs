use std::ops::RangeBounds;
use std::sync::Arc;

use crate::{Context, DeviceContext, Error};
use crate::context::VprContext;
use crate::image::{Frame, ImageView};

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
/// The before-mentioned categories of locality are as follows:
///
/// - Shared: Shared amongst all decoder instances on a device and amongst all
/// 		  cores in the system.
/// - Instance: Resources local to a specific decoder on a given device and
///             shared between frame production procedure calls.
/// - Worker: Resources shared between frames, but local to a single specific
/// 		  core in the system.
/// - Frame: Resources local to video frame production and pegged to specific
///          cores in the system.
///
pub trait Decoder {
	/// State data shared between all grouped instances of a decoder type.
	type SharedState;
	/// State data local to one decoder instance of a decoder type.
	type InstanceState;
	/// State data associated with each worker of the decoder.
	type WorkerState: ?Send + ?Sync;
	/// State data associated with each frame of the decoder.
	type FrameState: ?Send + ?Sync;
	/// Parameters passed by the scheduler function to the decoder function.
	type FrameParam;
	/// Error type used by this decoder.
	type Error;

	/// Consume raw encoded data and schedule frames for decoding
	fn schedule(&self,
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &mut Self::InstanceState,
		frames: &DecodeScheduler<Self>,
		data: &[u8]) -> Result<(), Self::Error>;

	/// Perform the decoding of the frame.
	fn decode(&self,
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		worker: &mut Self::WorkerState,
		frame: &mut Frame<Self::FrameState>,
		param: Self::FrameParam,
		data: &[u8]);

	fn create_shared_state(
		context: &DeviceContext) -> Result<Self::SharedState, Self::Error>;
	fn create_instance_state(
		context: &DeviceContext,
		shared: &Self::SharedState) -> Result<Self::InstanceState, Self::Error>;
	fn create_worker_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState) -> Result<Self::WorkerState, Self::Error>;
	fn create_frame_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		worker: &mut Self::WorkerState,
		image: &ImageView) -> Result<Self::FrameState, Self::Error>;

	fn destroy_shared_state(
		context: &DeviceContext,
		shared: &mut Self::SharedState);
	fn destroy_instance_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &mut Self::InstanceState);
	fn destroy_worker_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		worker: &mut Self::WorkerState);
	fn destroy_frame_state(
		context: &DeviceContext,
		shared: &Self::SharedState,
		instance: &Self::InstanceState,
		worker: &mut Self::WorkerState,
		frame: &mut Self::FrameState);
}

/// The decode queue.
///
/// This is the main structure through which decoding is actually performed and
/// decoding operations are set up.
pub struct DecodeQueue<C> {
	context: Arc<VprContext>,
	decoder: C,
	incoming_data_buffer: Vec<u8>,
}
impl<C> DecodeQueue<C> {
	pub(crate) fn new(
		context: Arc<VprContext>,
		decoder: C) -> Result<Self, Error> {

		Ok(Self {
			context,
			decoder,
			incoming_data_buffer: vec![]
		})
	}
}

/// The scheduler for frame decode operations.
pub struct DecodeScheduler<'a, C>
	where C: Decoder {

	source: &'a [u8],
	frames: Vec<(&'a [u8], C::FrameParam)>,
	_bind: std::marker::PhantomData<C>,
}
impl<C> DecodeScheduler<C>
	where C: Decoder {

	pub fn schedule<R>(
		&mut self,
		data: R,
		param: C::FrameParam)
		where R: RangeBounds<usize> {

		self.frames.push((&self.source[data], param))
	}
}
