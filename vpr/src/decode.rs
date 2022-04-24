use std::sync::Arc;
use crate::{Context, Error};

pub trait Decoder {

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
