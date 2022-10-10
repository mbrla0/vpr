
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Endianness {
	Big,
	Little,
}

/// Query the endianness of the host system.
///
pub const fn host_endianness() -> Result<Endianness, UnsupportedEndianness> {
	let n = [0x12, 0x34, 0x56, 0x78];
	let n = u32::from_ne_bytes(n);

	Ok(match n {
		0x12345678 => Endianness::Big,
		0x78563412 => Endianness::Little,
		_ => return Err(UnsupportedEndianness)
	})
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, thiserror::Error)]
#[error("The endianness of this system is not supported.")]
pub struct UnsupportedEndianness;
