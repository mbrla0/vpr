/// Instances recycling queue-related functions for all descriptor set types
/// the worker state may wish to distinguish between.
macro_rules! instance_descriptor_set_functions {
	($(
		pub unsafe (
			$(#[$getter_meta:meta])*
			fn $getter:ident,
			$(#[$ret_meta:meta])*
			fn $ret:ident
		) => $queue:ident;
	)*) => {$(
		$(#[$getter_meta])*
		pub unsafe fn $getter(
			&mut self,
			context: &DeviceContext,
			shared: &SharedState,
		) -> VkResult<::ash::vk::DescriptorSet> {
			let vk = context.device();
			if let Some(buffer) = self.$queue.pop_front() {
				vk.reset_command_buffer(
					buffer,
					::ash::vk::CommandBufferResetFlags::empty())?;
				Ok(buffer)
			} else {
				let buffers = vk.allocate_command_buffers(
					&::ash::vk::CommandBufferAllocateInfo::builder()
						.command_pool(self.command_buffer_pool)
						.command_buffer_count(1)
						.build())?;
				Ok(buffers[0])
			}
		}

		$(#[$ret_meta])*
		pub unsafe fn $ret(&mut self, set: vk::DescriptorSet) {
			self.$queue.push_back(set)
		}
	)*}
}

/// Creates a new compute pipeline from the given parameters.
macro_rules! create_compute_pipeline {
	(
		device: $vk:expr,
		cache: $cache:expr,
		layout: $layout:expr,
		module: $shader:expr,
		entry: $entry:expr,
		specialization: [
			$({
				index: $s_index:expr,
				offset: $s_offset:expr,
				data: $s_data:expr$(,)?
			}),*
		]$(,)?
	) => {{
		let mut specialization = vec![
			$({
				($s_index, $s_offset, $s_data)
			},)*
		];

		let buffer_size = specialization
			.iter()
			.map(|(_, offset, data)| offset + data.len())
			.max()
			.unwrap_or(0);
		let mut buffer = vec![0u8; buffer_size];
		let mut entries = ::alloc::Vec::with_capacity(specialization.len());

		for (index, offset, data) in specializations {
			entries.push(::ash::vk::SpecializationMapEntry::builder()
				.constant_id(index)
				.offset(offset));
			buffer[offset..offset + data.len()].copy_from_slice(data);
		}

		$vk.create_compute_pipelines(
				$cache,
				&[
					::ash::vk::ComputePipelineCreateInfo::builder()
						.stage(::ash::vk::PipelineShaderStageCreateInfo::builder()
							.stage(::ash::vk::ShaderStageFlags::COMPUTE)
							.module($shader)
							.name(::c_str::c_str!($entry))
							.specialization_info(&::ash::vk::SpecializationInfo::builder()
								.map_entries(&entries)
								.data(&buffer)
								.build())
							.build())
						.layout($layout)
						.build()
				],
				None)
				.map(|vector| vector[0])
	}}
}

pub use instance_descriptor_set_functions;
pub use create_compute_pipeline;