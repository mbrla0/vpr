#![feature(proc_macro_span)]
use proc_macro::TokenStream;
use syn::AttributeArgs;
use crate::item::ItemDeclaration;
mod request;
mod item;
mod compiler;

#[proc_macro_attribute]
pub fn include(attribute: TokenStream, item: TokenStream) -> TokenStream {
	let request = syn::parse_macro_input!(attribute as AttributeArgs);
	let request = request::parse(request);

	let item = syn::parse_macro_input!(item as ItemDeclaration);
	let ItemDeclaration { vis, memory, ident, .. } = item;

	let data = compiler::execute(&request).into_iter();
	quote::quote! {
		#vis #memory #ident: &'static [u32] = &[
			#(#data),*
		];
	}.into()
}