use quote::ToTokens;
use syn::{Attribute, Ident, Token, Type, Visibility};
use proc_macro2::TokenStream;
use syn::parse::{Parse, ParseStream};

pub struct ItemDeclaration {
	pub attrs: Vec<Attribute>,
	pub vis: Visibility,
	pub memory: Memory,
	pub ident: Ident,
	pub colon: Token![:],
	pub ty: Box<Type>,
	pub semicolon: Token![;],
}
impl Parse for ItemDeclaration {
	fn parse(input: ParseStream) -> syn::Result<Self> {
		Ok(Self {
			attrs: input.call(Attribute::parse_outer)?,
			vis: input.parse()?,
			memory: input.parse()?,
			ident: input.parse()?,
			colon: input.parse()?,
			ty: input.parse()?,
			semicolon: input.parse()?
		})
	}
}

pub enum Memory {
	Const(Token![const]),
	Static(Token![static]),
}
impl Parse for Memory {
	fn parse(input: ParseStream) -> syn::Result<Self> {
		if input.peek(Token![const]) {
			Ok(Self::Const(input.parse()?))
		} else if input.peek(Token![static]) {
			Ok(Self::Static(input.parse()?))
		} else {
			Err(syn::Error::new(
				input.span(),
				"expected either \"static\" or \"const\""))
		}
	}
}
impl ToTokens for Memory {
	fn to_tokens(&self, tokens: &mut TokenStream) {
		match self {
			Self::Static(st) => st.to_tokens(tokens),
			Self::Const(co) => co.to_tokens(tokens),
		}
	}
}