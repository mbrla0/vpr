use vpr::ProRes;

#[test]
fn prores_frame() {
	let instance = vpr::Instance::new(|devices| {
		devices[0].select();
	}).unwrap();

	let decoder = instance.decoder(ProRes).unwrap();

}
