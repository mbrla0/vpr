use std::io::Read;
use std::path::PathBuf;
use quote::ToTokens;
use syn::{AttributeArgs, Lit, Meta, NestedMeta};

pub fn parse(arguments: AttributeArgs) -> Request {
	let mut source = None;
	let mut stage = None;
	let mut target = None;

	for argument in arguments {
		match argument {
			NestedMeta::Meta(meta) => match meta {
				Meta::NameValue(kv) => {
					let key = kv.path
						.segments
						.to_token_stream()
						.to_string();
					let val = match kv.lit {
						Lit::Str(string) => string.value(),
						_ =>
							panic!("expected a string literal for the value \
								of \"{}\"", key)
					};

					match key.as_str() {
						"file" => {
							let old = source.replace(Source::File(val));
							if let Some(_) = old {
								panic!("multiple shader sources")
							}
						},
						"stage" => {
							let val = match val.as_str() {
								"vertex" => Stage::Vertex,
								"fragment" => Stage::Fragment,
								"tesscontrol" => Stage::TesselationControl,
								"tesseval" => Stage::TesselationEvaluation,
								"geometry" => Stage::Geometry,
								"compute" => Stage::Compute,
								what => panic!("unknown shader stage: {}", what)
							};
							let old = stage.replace(val);
							if let Some(_) = old {
								panic!("multiple shader stages")
							}
						},
						"target" => {
							let val = match val.as_str() {
								"vulkan" | "vulkan1.0" => VulkanTarget::Vulkan10,
								"vulkan1.1" => VulkanTarget::Vulkan11,
								"vulkan1.2" => VulkanTarget::Vulkan12,
								what => panic!("unknown vulkan target: {}", what)
							};
							let old = target.replace(val);
							if let Some(_) = old {
								panic!("multiple vulkan targets")
							}
						},
						what =>
							panic!("unrecognized parameter \"{}\"", what)
					}
				},
				Meta::List(list) =>
					panic!("expected attribute value pair, got list instead: {}",
						list.to_token_stream().to_string()),
				Meta::Path(path) =>
					panic!("expected attribute value pair, got path instead: {}",
						path.to_token_stream().to_string()),
			},
			NestedMeta::Lit(literal) =>
				panic!("expected attribute value pair, got literal instead: {}",
					literal.to_token_stream().to_string())
		}
	}

	let target = match target { Some(target) => target, None => VulkanTarget::Vulkan11 };
	let stage = match stage { Some(stage) => stage, None => panic!("must specify stage") };
	let source = match source { Some(source) => source, None => panic!("must specify source") };

	Request { target, stage, source }
}

pub struct Request {
	source: Source,
	stage: Stage,
	target: VulkanTarget
}
impl Request {
	pub fn load_source(&self) -> String {
		match &self.source {
			Source::Inline(source) => source.to_owned(),
			Source::File(path) => {
				let base = proc_macro::Span::call_site()
					.source_file()
					.path();
				let path = base
					.parent()
					.unwrap()
					.join(path);

				let mut file = match std::fs::File::open(&path) {
					Ok(file) => file,
					Err(what) => panic!("could not open file {:?}: {}", path, what)
				};

				let mut source = String::new();
				if let Err(what) = file.read_to_string(&mut source) {
					panic!("could not read from file {:?}: {}", path, what)
				}
				source
			}
		}
	}

	pub fn stage(&self) -> shaderc::ShaderKind {
		match self.stage {
			Stage::Vertex => shaderc::ShaderKind::Vertex,
			Stage::Fragment => shaderc::ShaderKind::Fragment,
			Stage::Compute => shaderc::ShaderKind::Compute,
			Stage::TesselationControl => shaderc::ShaderKind::TessControl,
			Stage::TesselationEvaluation => shaderc::ShaderKind::TessEvaluation,
			Stage::Geometry => shaderc::ShaderKind::Geometry,
			Stage::RayGeneration => shaderc::ShaderKind::RayGeneration,
			Stage::RayHit => shaderc::ShaderKind::AnyHit,
		}
	}

	pub fn target(&self) -> u32 {
		self.target.as_env_version()
	}

	pub fn source_path(&self) -> Option<PathBuf> {
		let path = match &self.source {
			Source::File(path) => path,
			Source::Inline(_) => return None
		};
		let base = proc_macro::Span::call_site()
			.source_file()
			.path();
		let path = base
			.parent()
			.unwrap()
			.join(path);
		let path = match path.canonicalize() {
			Ok(path) => path,
			Err(what) => panic!("could not find canonical path for source file \
				{:?}: {}", path, what)
		};
		Some(path)
	}
}

enum VulkanTarget {
	Vulkan10,
	Vulkan11,
	Vulkan12,
	Vulkan13,
}
impl VulkanTarget {
	pub fn as_env_version(&self) -> u32 {
		match self {
			VulkanTarget::Vulkan10 => shaderc::EnvVersion::Vulkan1_0 as u32,
			VulkanTarget::Vulkan11 => shaderc::EnvVersion::Vulkan1_1 as u32,
			VulkanTarget::Vulkan12 => shaderc::EnvVersion::Vulkan1_2 as u32,
			VulkanTarget::Vulkan13 => (1 << 22) | (3 << 12),
		}
	}
}

enum Stage {
	Vertex,
	Fragment,
	Compute,
	TesselationControl,
	TesselationEvaluation,
	Geometry,
	RayGeneration,
	RayHit,
}

enum Source {
	File(String),
	Inline(String),
}
