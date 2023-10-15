//! ProRes bitstream parser.
//!
//! This module defines parsers for the data structures in the ProRes video
//! stream. These structures are designed as to keep data copying to a minimum,
//! by striving to only keep local copies of metadata.
//!

use byteorder::{BigEndian, ReadBytesExt};

#[derive(Debug)]
#[repr(transparent)]
pub struct Frame<'a>(&'a [u8]);
impl<'a> Frame<'a> {
	const FIXED_FIELDS_SIZE: usize = 8;

	pub fn new(data: &'a [u8]) -> Option<Self> {
		if data.len() < Self::FIXED_FIELDS_SIZE {
			/* Can't even start to work on this. */
			return None
		}
		Some(Self(data))
	}

	/// Size of the frame, in bytes.
	pub fn len(&self) -> usize {
		let mut cursor = self.0;
		cursor.read_u32::<BigEndian>()
			.unwrap()
			.try_into()
			.unwrap()
	}

	/// Signature value of the frame.
	pub fn sig(&self) -> [u8; 4] {
		self.0[4..].split_array_ref::<4>().0.clone()
	}

	/// Frame header.
	pub fn header(&self) -> FrameHeader<'a> {
		FrameHeader(&self.0[Self::FIXED_FIELDS_SIZE..])
	}

	pub fn picture0(&self) -> Picture<'a> {
		Picture(&self.0[Self::FIXED_FIELDS_SIZE + self.header().len()..])
	}
}

#[allow(dead_code)]
#[derive(Debug)]
#[repr(transparent)]
pub struct FrameHeader<'a>(&'a [u8]);
impl<'a> FrameHeader<'a> {
	const FIXED_FIELDS_SIZE: usize = 20;
	const QUANTIZATION_MATRIX_SIZE: usize = 64;

	pub fn has_minimum_len(&self) -> bool {
		self.len() >= Self::FIXED_FIELDS_SIZE
	}

	pub fn len(&self) -> usize {
		let mut cursor = self.0;
		cursor.read_u16::<BigEndian>()
			.unwrap()
			.into()
	}

	pub fn reserved0(&self) -> u8 {
		self.0[2]
	}

	pub fn bitstream_version(&self) -> u8 {
		self.0[3]
	}

	pub fn encoder(&self) -> [u8; 4] {
		self.0[4..8].try_into().unwrap()
	}

	pub fn horizontal_size(&self) -> u16 {
		let mut cursor = &self.0[8..];
		cursor.read_u16::<BigEndian>().unwrap()
	}

	pub fn vertical_size(&self) -> u16 {
		let mut cursor = &self.0[10..];
		cursor.read_u16::<BigEndian>().unwrap()
	}

	pub fn chroma_format(&self) -> u8 {
		self.0[12] >> 6 & 3
	}

	pub fn reserved1(&self) -> u8 {
		self.0[12] >> 4 & 3
	}

	pub fn interlace_mode(&self) -> u8 {
		self.0[12] >> 2 & 3
	}

	pub fn reserved2(&self) -> u8 {
		self.0[12] & 3
	}

	pub fn aspect_ratio_mode(&self) -> u8 {
		self.0[13] >> 4 & 7
	}

	pub fn framerate_mode(&self) -> u8 {
		self.0[13] & 7
	}

	pub fn color_primaries(&self) -> u8 {
		self.0[14]
	}

	pub fn transfer_characteristic(&self) -> u8 {
		self.0[15]
	}

	pub fn matrix_coefficients(&self) -> u8 {
		self.0[16]
	}

	pub fn reserved3(&self) -> u8 {
		self.0[17] >> 4 & 7
	}

	pub fn alpha_channel_type(&self) -> u8 {
		self.0[17] & 7
	}

	pub fn reserved4(&self) -> u16 {
		let mut cursor = self.0;
		cursor.read_u16::<BigEndian>()
			.unwrap()
			.into() >> 2 & 0x3fff
	}

	pub fn has_luma_quantization_matrix(&self) -> bool {
		self.0[19] >> 1 & 1 == 1
	}

	pub fn has_chroma_quantization_matrix(&self) -> bool {
		self.0[19] & 1 == 0
	}

	pub fn luma_quantization_matrix(&self) -> Option<&'a [u8; Self::QUANTIZATION_MATRIX_SIZE]> {
		if !self.has_luma_quantization_matrix() {
			return None
		}

		let beg = Self::FIXED_FIELDS_SIZE;
		let end = Self::FIXED_FIELDS_SIZE + Self::QUANTIZATION_MATRIX_SIZE;
		self.0.get(beg..end)
			.try_into()
			.unwrap()
	}

	pub fn chroma_quantization_matrix(&self) -> Option<&'a [u8; Self::QUANTIZATION_MATRIX_SIZE]> {
		if !self.has_chroma_quantization_matrix() {
			return None
		}

		let off = if self.has_luma_quantization_matrix() { Self::QUANTIZATION_MATRIX_SIZE } else { 0 };

		let beg = Self::FIXED_FIELDS_SIZE + off;
		let end = Self::FIXED_FIELDS_SIZE + Self::QUANTIZATION_MATRIX_SIZE + off;
		self.0.get(beg..end)
			.try_into()
			.unwrap()
	}
}

pub struct Picture<'a>(&'a [u8]);
impl<'a> Picture<'a> {
	pub fn header(&self) -> PictureHeader<'a> {
		PictureHeader(&self.0[0..])
	}

	pub fn read_slice_sizes(&self, mb_height: u16, slices_per_row: u16) -> impl Iterator<Item = (u16, u16, u16)> + 'a {
		let mut cursor = &self.0[PictureHeader::FIXED_FIELDS_SIZE..];
		(0..mb_height)
			.flat_map(|y| std::iter::repeat(y).zip(0..slices_per_row))
			.map(move |(y, x)| (x, y, cursor.read_u16::<BigEndian>().unwrap()))
	}
}

pub struct PictureHeader<'a>(&'a [u8]);
impl<'a> PictureHeader<'a> {
	const FIXED_FIELDS_SIZE: usize = 8;

	pub fn len(&self) -> usize {
		(self.0[0] >> 3 & 31).into()
	}

	pub fn reserved0(&self) -> u8 {
		self.0[0] & 7
	}

	pub fn picture_size(&self) -> u32 {
		let mut cursor = &self.0[1..];
		cursor.read_u32::<BigEndian>()
			.unwrap()
	}

	pub fn ignored0(&self) -> u16 {
		let mut cursor = &self.0[5..];
		cursor.read_u16::<BigEndian>()
			.unwrap()
	}

	pub fn reserved1(&self) -> u8 {
		self.0[7] >> 6 & 3
	}

	pub fn log2_desired_slice_size_in_macro_blocks(&self) -> u8 {
		self.0[7] >> 4 & 3
	}

	pub fn reserved2(&self) -> u8 {
		self.0[7] & 15
	}

	pub fn
}
