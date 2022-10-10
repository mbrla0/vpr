//! ProRes bitstream syntax.
//!
//! This module defines parsers for the data structures in the ProRes video
//! stream. These structures are designed as to keep data copying to a minimum,
//! by striving to only keep local copies of metadata.
//!

struct Frame<'a> {
	frame_size: u32,
	frame_identifier: BitString<u32>,
	frame_header: FrameHeader,
	luma_quantization_matrix: Option<&'a QuantizationMatrix>,
	chroma_quantization_matrix: Option<&'a QuantizationMatrix>,
	first_picture: Picture<'a>,
	second_picture: Option<Picture<'a>>
}

struct FrameHeader {
	frame_header_size: u16,
	reserved: u8,
	bitstream_version: u8,
	encoder_identifier: BitString<u16>,
	horizontal_size: u16,
	vertical_size: u16,
	combo0: u8,
	combo1: u8,
	color_primaries: u8,
	transfer_characteristic: u8,
	matrix_coefficients: u8,
	combo2: u8,
	combo3: u8,
}

struct Picture<'a> {
	_a: std::marker::PhantomData<&'a ()>,
}

struct PictureHeader {
	combo0: u8,
	picture_size: u32,
	deprecated_number_of_slices: u16,
	combo2: u8,
}

/// A bit string housed by the given type.
///
/// The use of this type signals that, for the purpose of semantics, the data in
/// the inner type is only used for the purposes of bit storage, and that none
/// of its semantics should apply to the data being stored.
#[derive(Debug, Copy, Clone)]
struct BitString<T>(T);
impl<T> BitString<T> {
	/// Create a new bit string from the given value.
	///
	/// # Safety
	/// The caller must ensure that the type may be safely interpreted as a bit
	/// string. This is true of all numeric types.
	pub const unsafe fn new(value: T) -> Self {
		Self(value)
	}

	/// Copy the byte at the given offset without checking if the byte index
	/// is in bounds for the type being read.
	pub const unsafe fn byte_unchecked(&self, byte: usize) -> u8 {
		let bytes = &self.0 as *const _ as *const u8;
		bytes
			.offset(byte as isize)
			.read_unaligned()
	}

	/// Copy the bit at the given offset without checking if the bit index is
	/// in bounds for the type being read.
	pub unsafe fn bit_unchecked(&self, bit: usize) -> bool {
		(self.byte_unchecked(bit / 8) >> (bit % 8)) & 1 == 1
	}
}

#[repr(transparent)]
struct QuantizationMatrix([u8; 64]);
impl QuantizationMatrix {
	#[inline(always)]
	pub unsafe fn get_unchecked(&self, i: usize, j: usize) -> u8 {
		*self.0.get_unchecked(i * 8 + j)
	}

	/// The default matrix used for the luma component.
	///
	/// This quantization matrix shall be used when the implementation is
	/// instructed not to load a custom matrix from the frame header.
	pub const fn default_luma() -> Self {
		Self([4; 64])
	}
}
