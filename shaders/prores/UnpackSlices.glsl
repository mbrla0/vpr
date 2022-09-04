#version 460
#pragma shader_stage(compute)
#include "Constants.glsl"

/// UnpackSlices.glsl
///
/// aa

layout(constant_id = 0) const int cLittleEndian = 1;
layout(constant_id = 1) const int cSubsamplingMode = SUBSAMPLING_MODE_4_2_2;
layout(constant_id = 2) const int cAlphaFormat = ALPHA_CHANNEL_DISABLED;
layout(constant_id = 3) const int cScanningMode = SCANNING_MODE_PROGRESSIVE;

struct IndexEntry
{
	uint offset;
	uint position;
	uint coded_size;
};	

layout(set = 0, binding = 0, std430) restrict readonly buffer _FrameHeader
{
	uint quantization_matrices[32];

	IndexEntry slice_index[];
} gFrameHeader;

#define QUANTIZATION_LUMA 0
#define QUANTIZATION_CHROMA 16

#define QUANTIZATION(b, i, j) ((gFrameHeader.quantization_matrices[b + i + (j >> 2U)] >> ((j & 3U) * 8)) & 0xFFU)

layout(set = 0, binding = 1, std430) restrict readonly buffer _Frame
{
	uint blob[];
} gFrame;

layout(set = 1, binding = 0, rgba32f) restrict uniform image2D CoefficientImage;

#define INPUT_SLICE gFrameHeader.slice_index[gl_GlobalInvocationID.x]
#define INPUT_WORD(index) gFrame.blob[INPUT_SLICE.offset + (index)]

/// Patterns for block assembly in both progressive and interlaced modes.
const uint BLOCK_SCAN_INV[128] = uint[128](
	/* Progressive mode block assembly pattern. */
	0U,  1U,  8U,  9U,  2U,  3U, 10U, 11U,
	16U, 17U, 24U, 25U, 18U, 19U, 26U, 27U,
	 4U,  5U, 12U, 20U, 13U,  6U,  7U, 14U,
	21U, 28U, 29U, 22U, 15U, 23U, 30U, 31U,
	32U, 33U, 40U, 48U, 41U, 34U, 35U, 42U,
	49U, 56U, 57U, 50U, 43U, 36U, 37U, 44U,
	51U, 58U, 59U, 52U, 45U, 38U, 39U, 46U,
	53U, 60U, 61U, 54U, 47U, 55U, 62U, 63U,
	/* Interlaced mode block assembly pattern. */
	 0U,  8U,  1U,  9U, 16U, 24U, 17U, 25U,
	 2U, 10U,  3U, 11U, 18U, 26U, 19U, 27U,
	32U, 40U, 33U, 34U, 41U, 48U, 56U, 49U,
	42U, 35U, 43U, 50U, 57U, 58U, 51U, 59U,
	 4U, 12U,  5U,  6U, 13U, 20U, 28U, 21U,
	14U,  7U, 15U, 22U, 29U, 36U, 44U, 37U,
	30U, 23U, 31U, 38U, 45U, 52U, 60U, 53U,
	46U, 39U, 47U, 54U, 61U, 62U, 55U, 63U
);

#define BLOCK_SCAN_INV_PROGRESSIVE 0
#define BLOCK_SCAN_INV_INTERLACED 64

/// Patterns for macroblock assembly in both 4:4:4 and 4:2:2 modes.
const uint MACROBLOCK_SCAN_INV[8] = uint[8](
	/* 4:4:4 mode macroblock assembly pattern. */
	0U, 1U, 2U, 3U,
	/* 4:2:2 mode macroblock assembly pattern. */
	0U, 2U, 0U, 0U
);

#define MACROBLOCK_SCAN_INV_FULL 0
#define MACROBLOCK_SCAN_INV_HALF_HORZ 4

#define EXP_GOLOMB_CODE(a) (a << 8U)
#define RICE_EXP_COMBO_CODE(a, b, c) (0x01U | (a << 8U) | (b << 16U) | (c << 24U))

const uint CODEBOOK_ADAPTATION_TABLE[29] = uint[29](
	/* For the `difference` element, used by the DC coefficients. */
	EXP_GOLOMB_CODE(0),
	EXP_GOLOMB_CODE(0),
	RICE_EXP_COMBO_CODE(1, 2, 3),
	EXP_GOLOMB_CODE(0),
	/* For the `run` element, used by the AC coefficients. */
	RICE_EXP_COMBO_CODE(2, 0, 1),
	RICE_EXP_COMBO_CODE(2, 0, 1),
	RICE_EXP_COMBO_CODE(1, 0, 1),
	RICE_EXP_COMBO_CODE(1, 0, 1),
	EXP_GOLOMB_CODE(0),
	RICE_EXP_COMBO_CODE(1, 1, 2),
	RICE_EXP_COMBO_CODE(1, 1, 2),
	RICE_EXP_COMBO_CODE(1, 1, 2),
	RICE_EXP_COMBO_CODE(1, 1, 2),
	EXP_GOLOMB_CODE(1),
	EXP_GOLOMB_CODE(1),
	EXP_GOLOMB_CODE(1),
	EXP_GOLOMB_CODE(1),
	EXP_GOLOMB_CODE(1),
	EXP_GOLOMB_CODE(1),
	EXP_GOLOMB_CODE(1),
	/* For the `abs_level_minus_1` element, used by the AC coefficients. */
	RICE_EXP_COMBO_CODE(2, 0, 2),
	RICE_EXP_COMBO_CODE(1, 0, 1),
	RICE_EXP_COMBO_CODE(2, 0, 1),
	EXP_GOLOMB_CODE(0),
	EXP_GOLOMB_CODE(1),
	EXP_GOLOMB_CODE(1),
	EXP_GOLOMB_CODE(1),
	EXP_GOLOMB_CODE(1),
	EXP_GOLOMB_CODE(2)
);

#define CODEBOOK_ADAPTATION_DC_DIFFERENCE 0x00040000U
#define CODEBOOK_ADAPTATION_AC_RUN 0x00100004U
#define CODEBOOK_ADAPTATION_AC_ABS_LEVEL_MINUS_1 0x00090014U

/// Converts a symbol in the Golomb symbol alphabet to its corresponding integer
/// value.
///
/// This function inverts function used in ProRes that maps the signed integers
/// into the unsigned integers, so thtat they may be used in a standard version
/// of both the Golomb codes it makes heavy use of.
///
int golomb_symbol_to_int(uint symbol)
{
	int sign = int(symbol & 1U);
	int abs = int(symbol >> 1U);

	abs += 1;
	abs >>= 1;

	abs *= -sign * 2 + 1;
	return abs;
}

/// This structure keeps track of a bit-precise pointer into the input data.
struct BitCursor
{
	uint cached_word;
	uint word_offset;
	uint bit_offset;
};

/// This structure stores information about a cursor at a given point in time,
/// and allows tracking it as its state changes.
struct BitCursorCheckpoint
{
	uint word_offset;
	uint bit_offset;
};

/// Positions a given cursor at the given word and bit offset.
void cursor_seek(inout BitCursor cursor, uint word_index, uint bit_offset)
{
	cursor.word_offset = word_index;
	cursor.bit_offset = bit_offset;
	cursor.cached_word = INPUT_WORD(cursor.word_offset);
}

/// Moves a given cursor forward by the given number of bits.
void cursor_ffw(inout BitCursor cursor, uint bits)
{
	cursor.bit_offset += bits;
	if(cursor.bit_offset >= 32U)
	{
		cursor.word_offset += cursor.bit_offset >> 5U;
		cursor.bit_offset &= 31U;
		cursor.cached_word = INPUT_WORD(cursor.word_offset);
	}
}

/// Creates a new checkpoint of the given cursor.
///
/// The checkpoint will contain the state of the cursor as it was when the call
/// to this function was made. Later, if the user wishes to know how many bits
/// have been read since this checkpoint was taken, they may call the
/// cursor_distance() function.
void cursor_checkpoint(in BitCursor cursor, out BitCursorCheckpoint checkpoint)
{
	checkpoint.word_offset = cursor.word_offset;
	checkpoint.bit_offset = cursor.bit_offset;
}

/// Returns how many bits have been read since the creation of a checpoint.
///
/// This function doesn't concern itself with distances larger than 2^32 bits,
/// nor with cursors that have been seeked behind where they first were when
/// the checkpoint was taken. Its return value, therefore, musn't be taken into
/// account under those circumstances.
uint cursor_distance(in BitCursor cursor, in BitCursorCheckpoint checkpoint)
{
	uint words;
	if(checkpoint.word_offset > cursor.word_offset)
		return 0U;
	words = cursor.word_offset - checkpoint.word_offset;

	uint bits = (words << 5U) + cursor.bit_offset;
	if(checkpoint.bit_offset > bits)
		return 0U;

	return bits - checkpoint.bit_offset;
}

/// Pulls in the next bit from the input data, following the given cursor.
bool cursor_next_bit(inout BitCursor cursor)
{
	if(cursor.bit_offset >= 32U)
	{
		cursor.word_offset += cursor.bit_offset >> 5U;
		cursor.bit_offset &= 31U;

		cursor.cached_word = INPUT_WORD(cursor.word_offset);
	}

	uint bit_pos;
	if(cLittleEndian != 0)
	{
		/* The ProRes bitstream is naturally big endian and, so, we have to
		 * swap byte order as we pull the data in if it turns out that we're
		 * running in a little endian machine. */
		uint bit = cursor.bit_offset & 7U;
		uint byte = cursor.bit_offset >> 3U;

		byte = 3U - byte;
		bit_pos = (byte << 3U) | bit;
	}
	else
		/* We may just use natural bit oder, as the shader is running in a big
		 * endian machine. */
		bit_pos = cursor.bit_offset;

	bool set = ((cursor.cached_word >> cursor.bit_offset) & 1U) == 1U;
	cursor.bit_offset += 1;

	return set;
}

/// Pulls in the next number of bits from the cursor.
uint cursor_next_bits(inout BitCursor cursor, uint count)
{
	uint acc = 0;
	while(count > 0)
	{
		acc <<= 1;
		acc  |= cursor_next_bit(cursor) == true ? 1U : 0U;

		count -= 1;
	}
	return acc;
}

/// Parses the next Exponential Golomb code word with the given parameters,
/// following the given cursor.
int cursor_next_exp_golomb_code(
	inout BitCursor cursor,
	uint k)
{
	uint q = 0;
	while(!cursor_next_bit(cursor))
		q += 1;

	uint suffix = 1;
	for(uint i = 0; i < q + k; ++i)
	{
		suffix <<= 1;
		suffix |= cursor_next_bit(cursor) ? 1U : 0U;
	}

	uint factor = 1;
	factor <<= k;

	uint symbol = suffix - factor;
	return golomb_symbol_to_int(symbol);
}

/// Parses the next Exponential Golomb/Golomb-Rice Combination code word with
/// the given parameters, following the given cursor.
int cursor_next_rice_exp_combo_code(
	inout BitCursor cursor,
	uint threshold,
	uint k_rice,
	uint k_exp)
{
	uint q = 0;
	while(!cursor_next_bit(cursor))
		q += 1;

	if(q > threshold)
	{
		uint suffix = 1;
		for(uint i = 0; i < q + k_exp; ++i)
		{
			suffix <<= 1;
			suffix |= cursor_next_bit(cursor) ? 1U : 0U;
		}

		uint factor = 1;
		factor <<= k_exp;

		uint symbol = suffix - factor;
		symbol += (threshold + 1U) * (1U << k_rice);

		return golomb_symbol_to_int(symbol);
	}
	else
	{
		uint r = 0;
		for(uint i = 0; i < k_rice; ++i)
		{
			r <<= 1;
			r |= cursor_next_bit(cursor) ? 1U : 0U;
		}

		uint symbol = q * (1U << k_rice) + r;
		return golomb_symbol_to_int(symbol);
	}
}

/// Parses the next code word from the given codebook adaptation pattern.
///
/// Throughout the bitstream, pretty much all of the code words have the code
/// books used to decode them vary according to the previous value in a sequence
/// of decoded values.
///
/// This function accounts for that by parse the adaptation table at the
/// beggining of this file as well as the previous value in the sequence and
/// automatically determines which codebook should be used, as well as what
/// its parameters should be.
///
int cursor_next_adapted_code(
	inout BitCursor cursor,
	uint adaptation,
	int value)
{
	uint offset = adaptation  & 0xFFFFU;
	uint length = adaptation >> 16U;

	uint index = uint(value);
	if(value >= length)
		index = length - 1;

	uint params = CODEBOOK_ADAPTATION_TABLE[index];
	uint type = (params >> 0U) & 0xFFU;
	if(type == 0)
	{
		uint k = (params >> 8U) & 0xFFU;
		return cursor_next_exp_golomb_code(cursor, k);
	}
	else
	{
		uint threshold = (params >> 8U) & 0xFFU;
		uint k_rice = (params >> 16U) & 0xFFU;
		uint k_exp = (params >> 24U) & 0xFFU;
		return cursor_next_rice_exp_combo_code(cursor, threshold, k_rice, k_exp);
	}
}

bool end_of_data(
	in BitCursor cursor,
	in BitCursorCheckpoint begin,
	uint structure_size)
{
	uint bits = cursor_distance(cursor, begin);
	if(bits > structure_size)
		return true;
	uint rem = structure_size - bits;

	if(rem >= 32)
		return false;

	BitCursor c1 = cursor;

	bool pass = true;
	for(uint i = 0; i < rem; ++i)
		if(cursor_next_bit(c1))
		{
			pass = false;
			break;
		}

	return pass;
}

/// Derive the value of `qScale` for the given quantization index.
uint qscale(uint index)
{
	if(index <= 128)
		return index;
	else
		return 128 + ((index - 128) << 2U);
}

void unpack_coefficients(
	inout BitCursor cursor,
	uint coded_data_size,
	uvec2 base_mb_offset,
	uint log2_macroblock_count,
	uint log2_block_count_per_macroblock,
	uint macroblock_scan_pattern,
	uint block_scan_pattern,
	uint quantization_index,
	uint quantization_matrix,
	uint target_component)
{
	/* Keep track of how many bits we've read from the start of the structure. */
	BitCursorCheckpoint cp_beg;
	cursor_checkpoint(cursor, cp_beg);

	/* This macro will calculate the pixel coordinate into the target image for
	 * any given combination of macroblock, block and frequency bucket. */
	#define TARGET_PIXEL(mb, b, freq) \
		ivec2( \
			(MACROBLOCK_SCAN_INV[macroblock_scan_pattern + (b)]  & 1U) * 8 \
				+ (base_mb_offset + uvec2((mb), 0)).x * 16 \
				+ (BLOCK_SCAN_INV[block_scan_pattern + (freq)]  & 7U), \
			(MACROBLOCK_SCAN_INV[macroblock_scan_pattern + (b)] >> 1U) * 8 \
				+ (base_mb_offset + uvec2((mb), 0)).y * 16 \
				+ (BLOCK_SCAN_INV[block_scan_pattern + (freq)] >> 3U) \
		)
	#define TARGET_DEQUANTIZE_AND_STORE(mb, b, freq, val) \
		{ \
			ivec2 _pos = TARGET_PIXEL(mb, b, freq); \
			\
			float _dequantized = float(val) \
				* QUANTIZATION(quantization_matrix, (freq >> 3U), (freq & 7U))\
				* float(qscale(quantization_index)) \
				* 8.0; \
			\
			vec4 _value = imageLoad(CoefficientImage, _pos); \
			_value[target_component] = _dequantized; \
			imageStore(CoefficientImage, _pos, _value); \
 		}

	/* The first frequency index is encoded differentially. */
	int first_dc_coeff = cursor_next_exp_golomb_code(cursor, 5);
	int last_dc_diff = 3;
	int last_dc_coeff = first_dc_coeff;

	TARGET_DEQUANTIZE_AND_STORE(0, 0, 0, first_dc_coeff);
	uint fj = 1;
	for(uint i = 0; i < (1U << log2_macroblock_count); ++i)
	{
		uvec2 mb_offset = base_mb_offset + uvec2(i, 0);
		for (uint j = fj; j < (1U << log2_block_count_per_macroblock); ++j)
		{
			int diff = cursor_next_adapted_code(
				cursor,
				CODEBOOK_ADAPTATION_DC_DIFFERENCE,
				abs(last_dc_coeff));

			int val = last_dc_coeff + diff;
			if(last_dc_diff < 0)
				val = -val;

			TARGET_DEQUANTIZE_AND_STORE(i, j, 0, val);

			last_dc_diff = diff;
			last_dc_coeff = val;
		}
		fj = 0;
	}

	/* All of the frequency indices following the first are run-length encoded. */
	int last_ac_run = 4;
	int last_ac_level = 1;

	uint coefficient = (1U << (log2_macroblock_count + log2_block_count_per_macroblock));
	#define TOTAL_COEFFICIENTS (1U << (log2_macroblock_count + log2_block_count_per_macroblock + 6))
	#define COEFFICIENT_BLOCK (coefficient & ((2U << log2_block_count_per_macroblock) - 1U))
	#define COEFFICIENT_MB (coefficient >> log2_block_count_per_macroblock)
	#define COEFFICIENT_FREQ (coefficient >> (log2_macroblock_count + log2_block_count_per_macroblock))
	#define DEQUANTIZE_AND_STORE_NEXT_COEFFICIENT(val) \
		{ \
			if(coefficient >= TOTAL_COEFFICIENTS) break; \
			TARGET_DEQUANTIZE_AND_STORE(COEFFICIENT_MB, COEFFICIENT_BLOCK, COEFFICIENT_FREQ, val); \
			coefficient += 1; \
 		}

	while(!end_of_data(cursor, cp_beg, coded_data_size))
	{
		int run = cursor_next_adapted_code(
			cursor,
			CODEBOOK_ADAPTATION_AC_RUN,
			last_ac_run);
		for(uint i = 0; i < run; ++i)
			DEQUANTIZE_AND_STORE_NEXT_COEFFICIENT(0);

		int abs_level_minus_1 = cursor_next_adapted_code(
			cursor,
			CODEBOOK_ADAPTATION_AC_ABS_LEVEL_MINUS_1,
			last_ac_level);

		bool negative = cursor_next_bit(cursor);

		int level = (abs_level_minus_1 + 1) * (1 - 2 * (negative ? 1 : 0));
		DEQUANTIZE_AND_STORE_NEXT_COEFFICIENT(level);
	}

	/* The last run is implicit, fill all of the coefficients we haven't yet
	 * with zeroes. */
	while(coefficient < TOTAL_COEFFICIENTS)
		DEQUANTIZE_AND_STORE_NEXT_COEFFICIENT(0);

	uint covered = cursor_distance(cursor, cp_beg);
	if(covered <= (coded_data_size << 3U))
		cursor_ffw(cursor, (coded_data_size << 3U) - covered);
}

void main()
{
	BitCursor cursor;
	cursor_seek(cursor, 0, 0);

	/* Parse the slice's header and derive all of the quantities related to it. */
	uint slice_header_size = cursor_next_bits(cursor, 8) >> 3;
	uint quantization_index = cursor_next_bits(cursor, 8);
	uint coded_size_of_y_data = cursor_next_bits(cursor, 16);
	uint coded_size_of_cb_data = cursor_next_bits(cursor, 16);
	uint coded_size_of_cr_data;
	if(cAlphaFormat == ALPHA_CHANNEL_DISABLED)
		coded_size_of_cr_data = cursor_next_bits(cursor, 16);
	else
		coded_size_of_cr_data = 
			  INPUT_SLICE.coded_size
			- slice_header_size
			- coded_size_of_y_data
			- coded_size_of_cb_data;  
	
	uvec2 slice_mb_offset = uvec2(
		(INPUT_SLICE.position & 0x00007fffU) >> 0,
		(INPUT_SLICE.position & 0x3fff8000U) >> 15);
	uint log2_slice_len_mb = (INPUT_SLICE.position & 0xc0000000U) >> 30;

	/* Seek the cursor to the start of the coded color data. */
	cursor_seek(
		cursor, 
		slice_header_size >> 2U, 
		(slice_header_size & 3U) << 3U);

	/* Set the picture scanning modes. */
	uint chroma_mb_scan;
	uint block_scan;
	uint chroma_log2_block_count_per_macroblock;

	if(cSubsamplingMode == SUBSAMPLING_MODE_4_2_2)
	{
		chroma_mb_scan = MACROBLOCK_SCAN_INV_HALF_HORZ;
		chroma_log2_block_count_per_macroblock = 1;
	}
	else
	{
		chroma_mb_scan = MACROBLOCK_SCAN_INV_FULL;
		chroma_log2_block_count_per_macroblock = 2;
	}
	
	if(cScanningMode == SCANNING_MODE_INTERLACED)
		block_scan = BLOCK_SCAN_INV_INTERLACED;
	else
		block_scan = BLOCK_SCAN_INV_PROGRESSIVE;

	/* Unpack the coefficients for the luma and chroma information. */
	unpack_coefficients(
		cursor,
		coded_size_of_y_data,
		slice_mb_offset,
		log2_slice_len_mb,
		2,
		MACROBLOCK_SCAN_INV_FULL,
		block_scan,
		quantization_index,
		QUANTIZATION_LUMA,
		0);

	unpack_coefficients(
		cursor,
		coded_size_of_cb_data,
		slice_mb_offset,
		log2_slice_len_mb,
		chroma_log2_block_count_per_macroblock,
		chroma_mb_scan,
		block_scan,
		quantization_index,
		QUANTIZATION_CHROMA,
		1);

	unpack_coefficients(
		cursor,
		coded_size_of_cr_data,
		slice_mb_offset,
		log2_slice_len_mb,
		chroma_log2_block_count_per_macroblock,
		chroma_mb_scan,
		block_scan,
		quantization_index,
		QUANTIZATION_CHROMA,
		2);
}
