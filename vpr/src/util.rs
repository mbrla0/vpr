use ash::vk;

/// Pick the memory type with the largest heap that fits the given constraints.
pub fn pick_largest_memory_heap(
	memory_properties: &vk::PhysicalDeviceMemoryProperties,
	mask: u32,
	type_requirements: vk::MemoryPropertyFlags,
	heap_requirements: vk::MemoryHeapFlags,
) -> Option<u32> {
	let mut candidates = 0u32;
	for i in 0..memory_properties.memory_type_count {
		let memory_type = memory_properties.memory_types[i as usize];
		let memory_heap = memory_properties.memory_heaps[memory_type.heap_index as usize];

		let bit = 1u32 << i;
		if bit & mask == 0 { continue }

		let a = memory_type.property_flags.contains(type_requirements);
		let b = memory_heap.flags.contains(heap_requirements);
		if !a || !b { continue }

		candidates |= bit;
	}

	let mut chosen_type = None;
	for i in 0..u32::BITS {
		let is_candidate = (candidates >> i) & 1 != 0;
		if !is_candidate { continue }

		let memory_type = memory_properties
			.memory_types[i as usize];
		let memory_heap = memory_properties
			.memory_heaps[memory_type.heap_index as usize];

		if let Some(chosen_type) = chosen_type {
			let chosen_memory_type = memory_properties
				.memory_types[chosen_type as usize];
			let chosen_memory_heap = memory_properties
				.memory_heaps[chosen_memory_type.heap_index as usize];

			if chosen_memory_heap.size >= memory_heap.size {
				continue
			}
		}

		chosen_type = Some(i)
	}

	chosen_type
}
