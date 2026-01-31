#![no_std]

use core::ops::Neg;
use polynomial::{
    num_traits::{Num, Pow, Zero},
    Polynomial, PolynomialError,
};

pub use polynomial::{Poly0, Poly1, Poly2, Poly3, Poly4, Poly5, PolynomialConst};

#[derive(Clone, Debug, thiserror::Error)]
pub enum ZFilterError {
    #[error("Length error: index {} out of bounds", index)]
    LengthError { index: usize },
    #[error("Poly calc error")]
    PolyCalcError(#[from] PolynomialError),
}

pub trait PolyNormalize
where
    Self: Sized,
{
    type Norm;
    fn normalize(&self) -> Result<(Self, Self::Norm), ZFilterError>;
    fn normalize_by(&self, norm: Self::Norm) -> Result<Self, ZFilterError>;
}

pub trait GenerateInnerState<T: Copy, Inner>
where
    Inner: InnerState<T>,
{
    fn generate_inner_state() -> Inner;
}

pub trait InnerState<T> {
    type Iter<'a>: Iterator<Item = &'a T>
    where
        Self: 'a,
        T: 'a;
    type IterMut<'a>: Iterator<Item = &'a mut T>
    where
        Self: 'a,
        T: 'a;

    fn get(&self, idx: usize) -> Option<&T>;
    fn get_mut(&mut self, idx: usize) -> Option<&mut T>;
    fn iter(&self) -> Self::Iter<'_>;
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
    fn order(&self) -> usize;
    fn reset(&mut self)
    where
        T: Copy + Default,
    {
        for elem in self.iter_mut() {
            *elem = T::default();
        }
    }
}

impl<T, C> InnerState<T> for C
where
    C: AsRef<[T]> + AsMut<[T]> + Clone,
    T: Copy,
{
    type Iter<'a>
        = core::slice::Iter<'a, T>
    where
        Self: 'a,
        T: 'a;
    type IterMut<'a>
        = core::slice::IterMut<'a, T>
    where
        Self: 'a,
        T: 'a;

    fn get(&self, idx: usize) -> Option<&T> {
        self.as_ref().get(idx)
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        self.as_mut().get_mut(idx)
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.as_ref().iter()
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.as_mut().iter_mut()
    }

    fn order(&self) -> usize {
        self.as_ref().len()
    }
}

pub trait Filter<T: Copy> {
    fn reset(&mut self);
    fn process_sample(&mut self, x: T) -> T;
    #[inline]
    fn process_buffer(&mut self, x: &[T], y: &mut [T]) {
        assert!(x.len() == y.len());
        for (i, &xi) in x.iter().enumerate() {
            y[i] = self.process_sample(xi);
        }
    }
}

impl<T, const N: usize> PolyNormalize for PolynomialConst<T, N>
where
    T: Num + Copy + Zero + Default,
{
    type Norm = T;
    fn normalize(&self) -> Result<(Self, Self::Norm), ZFilterError> {
        let leading_coeff = self
            .coeff(0)
            .ok_or(ZFilterError::LengthError { index: 0 })?;
        Ok(((self.clone() / leading_coeff)?, leading_coeff))
    }

    fn normalize_by(&self, norm: Self::Norm) -> Result<Self, ZFilterError> {
        Ok((self.clone() / norm)?)
    }
}

impl<T, const N: usize> GenerateInnerState<T, [T; N]> for PolynomialConst<T, N>
where
    T: Copy + Default + Num,
{
    fn generate_inner_state() -> [T; N] {
        [T::default(); N]
    }
}

#[derive(Clone, Debug)]
pub struct DirectFormIITransposed<T, Poly, Inner>
where
    T: Num + Copy + Default,
    Poly: Polynomial<T> + PolyNormalize<Norm = T> + GenerateInnerState<T, Inner>,
    Inner: InnerState<T>,
{
    a: Poly,
    b: Poly,
    s: Inner,
}

impl<T, Poly, Inner> DirectFormIITransposed<T, Poly, Inner>
where
    T: Num + Copy + Default,
    Poly: Polynomial<T> + PolyNormalize<Norm = T> + GenerateInnerState<T, Inner>,
    Inner: InnerState<T>,
{
    pub fn new(a: Poly, b: Poly) -> Result<Self, ZFilterError> {
        let s = Poly::generate_inner_state();
        let (a, leading_coeff) = a.normalize()?;
        let b = b.normalize_by(leading_coeff)?;
        Ok(Self { a, b, s })
    }
}

pub fn bilinear_transform<T, Poly, Inner>(
    num_s: Poly,
    den_s: Poly,
    c: T, // = (2/Ts)*prewarp
) -> Result<DirectFormIITransposed<T, Poly, Inner>, ZFilterError>
where
    T: Num + Copy + Default + Pow<u8, Output = T> + Neg<Output = T>,
    Poly: Polynomial<T> + PolyNormalize<Norm = T> + GenerateInnerState<T, Inner>,
    Inner: InnerState<T>,
{
    let n = num_s.order().max(den_s.order());
    // Errorになるとき、0次であるのでDefaultで対応
    let one_minus_v = Poly::new(&[T::one(), -T::one()]).unwrap_or(Poly::default());
    let one_plus_v = Poly::new(&[T::one(), T::one()]).unwrap_or(Poly::default());
    let b = num_s
        .coeffs()
        .iter()
        .enumerate()
        .try_fold(Poly::default(), |acc, (k, &num_k)| {
            Ok::<_, ZFilterError>(
                acc + (one_minus_v.clone().pow(k as u8)?
                    * one_plus_v.clone().pow((n - k) as u8)?)?
                    * num_k
                    * c.pow(n as u8),
            )
        })?;
    let a = den_s
        .coeffs()
        .iter()
        .enumerate()
        .try_fold(Poly::default(), |acc, (k, &den_k)| {
            Ok::<_, ZFilterError>(
                acc + (one_minus_v.clone().pow(k as u8)?
                    * one_plus_v.clone().pow((n - k) as u8)?)?
                    * den_k
                    * c.pow(n as u8),
            )
        })?;
    DirectFormIITransposed::new(a, b)
}

impl<T, Poly, Inner> Filter<T> for DirectFormIITransposed<T, Poly, Inner>
where
    T: Num + Copy + Default + Zero,
    Poly: Polynomial<T> + PolyNormalize<Norm = T> + GenerateInnerState<T, Inner>,
    Inner: InnerState<T>,
{
    fn reset(&mut self) {
        self.s.reset();
    }

    fn process_sample(&mut self, x: T) -> T {
        let n = self.s.order();

        // y = b0 * x + s[0]
        let y =
            self.b.coeff(0).unwrap_or(T::zero()) * x + self.s.get(0).copied().unwrap_or(T::zero());

        // b_{k+1}, a_{k+1} を 1..=len で取り出し、足りない分は 0 でパディング
        let b_tail = (1..=n).map(|k| self.b.coeff(k).unwrap_or(T::zero()));
        let a_tail = (1..=n).map(|k| self.a.coeff(k).unwrap_or(T::zero()));

        // s[k] を前から順に更新する。
        // s[k] は「古い s[k+1]」を使う必要があるので、毎回「次の値」を先に読む。
        // s[k] = s[k+1] + b[k+1] * x - a[k+1] * y
        for (k, (bk1, ak1)) in b_tail.zip(a_tail).enumerate() {
            let next_old = self.s.get(k + 1).copied().unwrap_or(T::zero());
            if let Some(s_k) = self.s.get_mut(k) {
                *s_k = next_old + bk1 * x - ak1 * y;
            }
        }
        y
    }
}

pub type ZFilter1stOrder<T> = DirectFormIITransposed<T, Poly1<T>, [T; 2]>;
pub type ZFilter2ndOrder<T> = DirectFormIITransposed<T, Poly2<T>, [T; 3]>;
pub type ZFilter3rdOrder<T> = DirectFormIITransposed<T, Poly3<T>, [T; 4]>;
pub type ZFilter4thOrder<T> = DirectFormIITransposed<T, Poly4<T>, [T; 5]>;
pub type ZFilter5thOrder<T> = DirectFormIITransposed<T, Poly5<T>, [T; 6]>;
