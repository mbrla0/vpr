use std::path::Path;
use shaderc::{IncludeType, ResolvedInclude, SourceLanguage, TargetEnv};
use crate::request::Request;

pub fn execute(request: &Request) -> Vec<u32> {
	let source = request.load_source();
	let mut compiler = match shaderc::Compiler::new() {
		Some(compiler) => compiler,
		None => panic!("cannot initialize the glslc backend")
	};
	let source_path = request.source_path();

	let mut settings = shaderc::CompileOptions::new().unwrap();
	settings.set_target_env(TargetEnv::Vulkan, request.target());
	settings.set_source_language(SourceLanguage::GLSL);
	settings.set_include_callback(|path, kind, source, _| match (&source_path, kind) {
		(Some(_), IncludeType::Standard) => Ok({
			let parent = Path::new(source)
				.parent()
				.unwrap();
			let target = parent.join(path);

			let content = std::fs::read_to_string(&target)
				.map_err(|what| format!("could not read from file {:?}: {}", target, what))?;

			ResolvedInclude {
				resolved_name: target.to_str()
					.ok_or_else(|| format!("file name {:?} is not valid utf8", target))?
					.to_owned(),
				content
			}
		}),
		_ => Err("unsupported include type".to_owned())
	});
	let compiled = compiler.compile_into_spirv(
		&source,
		request.stage(),
		source_path.clone()
			.and_then(|name| name.to_str().map(|a| a.to_owned()))
			.unwrap_or("<inline>".to_owned())
			.as_str(),
		"main",
		Some(&settings)).unwrap();

	compiled.as_binary().to_owned()
}
