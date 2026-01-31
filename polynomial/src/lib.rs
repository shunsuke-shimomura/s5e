#![no_std]

use core::ops::{Add, Div, Mul, Sub};
use num_traits::{Num, Pow};

pub extern crate num_traits;

#[derive(thiserror::Error, Debug, Clone)]
pub enum PolynomialError {
    #[error("Polynomial order overflow")]
    OrderOverflow,
    #[error("Division by zero")]
    DivisionByZero,
}

#[allow(clippy::len_without_is_empty)]
pub trait Polynomial<T: Copy>:
    Sized
    + Default
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Result<Self, PolynomialError>>
    + Pow<u8, Output = Result<Self, PolynomialError>>
    + Mul<T, Output = Self>
    + Div<T, Output = Result<Self, PolynomialError>>
{
    fn new(coeffs: &[T]) -> Result<Self, PolynomialError>;
    fn order(&self) -> usize; // N
    fn coeff(&self, i: usize) -> Option<T>; // a_i, i=0..N
    fn coeffs(&self) -> &[T];
    #[inline]
    fn len(&self) -> usize {
        self.order() + 1
    }

    #[inline]
    fn eval_horner(&self, x: T) -> T
    where
        T: Num,
    {
        let mut acc = T::zero();
        for i in (0..self.len()).rev() {
            acc = acc * x + self.coeff(i).unwrap_or_else(|| T::zero());
        }
        acc
    }
}

#[derive(Clone, Debug)]
pub struct PolynomialConst<T, const N: usize>
where
    T: Copy + Default + Num,
{
    coeffs: [T; N],
}

impl<T, const N: usize> Default for PolynomialConst<T, N>
where
    T: Copy + Default + Num,
{
    fn default() -> Self {
        PolynomialConst {
            coeffs: [T::default(); N],
        }
    }
}

impl<T, const N: usize> Polynomial<T> for PolynomialConst<T, N>
where
    T: Copy + Default + Num,
{
    #[inline]
    fn order(&self) -> usize {
        N - 1
    }
    #[inline]
    fn coeff(&self, i: usize) -> Option<T> {
        self.coeffs.get(i).copied()
    }
    #[inline]
    fn coeffs(&self) -> &[T] {
        self.coeffs.as_slice()
    }
    fn new(coeffs: &[T]) -> Result<Self, PolynomialError> {
        if coeffs.len() > N {
            return Err(PolynomialError::OrderOverflow);
        }
        let mut coeffs_array = [T::default(); N];
        coeffs_array[..coeffs.len()].copy_from_slice(coeffs);
        Ok(PolynomialConst {
            coeffs: coeffs_array,
        })
    }
}

impl<T, const N: usize> Add for PolynomialConst<T, N>
where
    T: Copy + Default + Num,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut coeffs = [T::zero(); N];
        self.coeffs()
            .iter()
            .zip(rhs.coeffs().iter())
            .zip(coeffs.iter_mut())
            .for_each(|((c1, c2), c_out)| {
                *c_out = *c1 + *c2;
            });
        PolynomialConst { coeffs }
    }
}

impl<T, const N: usize> Sub for PolynomialConst<T, N>
where
    T: Copy + Default + Num,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut coeffs = [T::zero(); N];
        self.coeffs()
            .iter()
            .zip(rhs.coeffs().iter())
            .zip(coeffs.iter_mut())
            .for_each(|((c1, c2), c_out)| {
                *c_out = *c1 - *c2;
            });
        PolynomialConst { coeffs }
    }
}

impl<T, const N: usize> Mul for PolynomialConst<T, N>
where
    T: Copy + Default + Num,
{
    type Output = Result<Self, PolynomialError>;

    fn mul(self, rhs: Self) -> Result<Self, PolynomialError> {
        let mut coeffs = [T::zero(); N];
        self.coeffs().iter().enumerate().try_for_each(|(i, a)| {
            rhs.coeffs().iter().enumerate().try_for_each(|(j, b)| {
                let res = *a * *b;
                if let Some(c) = coeffs.get_mut(i + j) {
                    *c = *c + res;
                } else if res != T::zero() {
                    return Err(PolynomialError::OrderOverflow);
                }
                Ok(())
            })
        })?;
        Ok(PolynomialConst { coeffs })
    }
}

impl<T, const N: usize> Pow<u8> for PolynomialConst<T, N>
where
    T: Copy + Default + Num,
{
    type Output = Result<Self, PolynomialError>;

    fn pow(self, k: u8) -> Result<Self, PolynomialError> {
        let mut init = [T::zero(); N];
        init[0] = T::one();
        let init_poly = PolynomialConst { coeffs: init };

        (0..k).try_fold(init_poly, |acc, _| acc * self.clone())
    }
}

impl<T, const N: usize> Mul<T> for PolynomialConst<T, N>
where
    T: Copy + Default + Num,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        let mut coeffs = [T::zero(); N];
        self.coeffs()
            .iter()
            .zip(coeffs.iter_mut())
            .for_each(|(c1, c_out)| {
                *c_out = *c1 * rhs;
            });
        PolynomialConst { coeffs }
    }
}

impl<T, const N: usize> Div<T> for PolynomialConst<T, N>
where
    T: Copy + Default + Num,
{
    type Output = Result<Self, PolynomialError>;
    fn div(self, rhs: T) -> Self::Output {
        if rhs.is_zero() {
            return Err(PolynomialError::DivisionByZero);
        }
        let mut coeffs = [T::zero(); N];
        self.coeffs()
            .iter()
            .zip(coeffs.iter_mut())
            .for_each(|(c1, c_out)| {
                *c_out = *c1 / rhs;
            });
        Ok(PolynomialConst { coeffs })
    }
}

impl<T, const N: usize> PolynomialConst<T, N>
where
    T: Copy + Default + Num,
{
    pub fn new(coeffs: [T; N]) -> Self {
        PolynomialConst { coeffs }
    }

    pub fn pad_order<const M: usize>(&self) -> Result<PolynomialConst<T, M>, PolynomialError> {
        if M < N {
            return Err(PolynomialError::OrderOverflow);
        }
        let mut coeffs = [T::zero(); M];
        coeffs[..self.order() + 1].copy_from_slice(self.coeffs());
        Ok(PolynomialConst { coeffs })
    }
}

pub type Poly0<T> = PolynomialConst<T, 1>;
pub type Poly1<T> = PolynomialConst<T, 2>;
pub type Poly2<T> = PolynomialConst<T, 3>;
pub type Poly3<T> = PolynomialConst<T, 4>;
pub type Poly4<T> = PolynomialConst<T, 5>;
pub type Poly5<T> = PolynomialConst<T, 6>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_creation() {
        // Test creating a polynomial with new()
        let p1 = Poly1::<i32>::new([1, 2]); // 1 + 2x
        assert_eq!(p1.order(), 1);
        assert_eq!(p1.coeff(0), Some(1));
        assert_eq!(p1.coeff(1), Some(2));

        let p2 = Poly2::<f32>::new([1.0, 0.0, 3.0]); // 1 + 0x + 3x^2
        assert_eq!(p2.order(), 2);
        assert_eq!(p2.coeff(0), Some(1.0));
        assert_eq!(p2.coeff(1), Some(0.0));
        assert_eq!(p2.coeff(2), Some(3.0));
    }

    #[test]
    fn test_polynomial_evaluation() {
        // Test Horner's method evaluation
        let p = Poly2::<i32>::new([1, 2, 3]); // 1 + 2x + 3x^2

        // At x=0: 1 + 2*0 + 3*0^2 = 1
        assert_eq!(p.eval_horner(0), 1);

        // At x=1: 1 + 2*1 + 3*1^2 = 6
        assert_eq!(p.eval_horner(1), 6);

        // At x=2: 1 + 2*2 + 3*2^2 = 1 + 4 + 12 = 17
        assert_eq!(p.eval_horner(2), 17);
    }

    #[test]
    fn test_polynomial_addition() {
        // Test adding two polynomials
        let p1 = Poly1::<i32>::new([1, 2]); // 1 + 2x
        let p2 = Poly1::<i32>::new([3, 4]); // 3 + 4x
        let result = p1 + p2; // 4 + 6x

        assert_eq!(result.coeff(0), Some(4));
        assert_eq!(result.coeff(1), Some(6));

        // Test adding polynomials of different orders
        let p3 = Poly1::<i32>::new([1, 2]); // 1 + 2x
        let p4 = Poly2::<i32>::new([3, 4, 5]); // 3 + 4x + 5x^2
        let result2 = p3.pad_order().unwrap() + p4; // 4 + 6x + 5x^2

        assert_eq!(result2.coeff(0), Some(4));
        assert_eq!(result2.coeff(1), Some(6));
        assert_eq!(result2.coeff(2), Some(5));
    }

    #[test]
    fn test_polynomial_subtraction() {
        // Test subtracting two polynomials
        let p1 = Poly1::<i32>::new([5, 7]); // 5 + 7x
        let p2 = Poly1::<i32>::new([2, 3]); // 2 + 3x
        let result = p1 - p2; // 3 + 4x

        assert_eq!(result.coeff(0), Some(3));
        assert_eq!(result.coeff(1), Some(4));
    }

    #[test]
    fn test_polynomial_multiplication() {
        // Test multiplying two polynomials
        let p1 = Poly1::<i32>::new([1, 2]).pad_order().unwrap(); // 1 + 2x
        let p2 = Poly1::<i32>::new([3, 4]).pad_order().unwrap(); // 3 + 4x
        let result: Poly2<i32> = (p1 * p2).unwrap(); // (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2

        assert_eq!(result.coeff(0), Some(3)); // 1*3
        assert_eq!(result.coeff(1), Some(10)); // 1*4 + 2*3
        assert_eq!(result.coeff(2), Some(8)); // 2*4
    }

    #[test]
    fn test_polynomial_scalar_multiplication() {
        // Test multiplying polynomial by scalar
        let p = Poly2::<i32>::new([1, 2, 3]); // 1 + 2x + 3x^2
        let result = p * 2; // 2 + 4x + 6x^2

        assert_eq!(result.coeff(0), Some(2));
        assert_eq!(result.coeff(1), Some(4));
        assert_eq!(result.coeff(2), Some(6));
    }

    #[test]
    fn test_polynomial_pad_coeffs() {
        // Test padding coefficients to higher order
        let p1 = Poly1::<i32>::new([1, 2]); // 1 + 2x
        let p3: Poly3<_> = p1.pad_order().unwrap(); // 1 + 2x + 0x^2 + 0x^3

        assert_eq!(p3.order(), 3);
        assert_eq!(p3.coeff(0), Some(1));
        assert_eq!(p3.coeff(1), Some(2));
        assert_eq!(p3.coeff(2), Some(0));
        assert_eq!(p3.coeff(3), Some(0));
    }

    #[test]
    fn test_polynomial_mul_fixed_order() {
        // Test fixed-order multiplication
        let p1 = Poly1::<i32>::new([1, 2]); // 1 + 2x
        let p2 = Poly1::<i32>::new([3, 4]); // 3 + 4x

        // This should fail because (1+2x)(3+4x) = 3+10x+8x^2 has order 2, but we're trying to fit in order 1
        assert!((p1 * p2).is_err());

        // Test with polynomials that don't overflow
        let p3 = Poly2::<i32>::new([1, 0, 0]); // 1
        let p4 = Poly2::<i32>::new([2, 0, 0]); // 2
        let result = (p3 * p4).unwrap(); // 2

        assert_eq!(result.coeff(0), Some(2));
        assert_eq!(result.coeff(1), Some(0));
        assert_eq!(result.coeff(2), Some(0));
    }

    #[test]
    fn test_polynomial_power() {
        // Test polynomial exponentiation
        let p = Poly1::<i32>::new([1, 1]); // 1 + x
        let result: Poly4<_> = p.pad_order().unwrap().pow(2).unwrap(); // (1 + x)^2 = 1 + 2x + x^2

        assert_eq!(result.coeff(0), Some(1)); // constant term
        assert_eq!(result.coeff(1), Some(2)); // x term
        assert_eq!(result.coeff(2), Some(1)); // x^2 term
        assert_eq!(result.coeff(3), Some(0)); // x^3 term
        assert_eq!(result.coeff(4), Some(0)); // x^4 term
    }

    #[test]
    fn test_polynomial_default() {
        // Test default polynomial creation
        let p = Poly2::<i32>::default(); // 0 + 0x + 0x^2

        assert_eq!(p.order(), 2);
        assert_eq!(p.coeff(0), Some(0));
        assert_eq!(p.coeff(1), Some(0));
        assert_eq!(p.coeff(2), Some(0));
    }

    #[test]
    fn test_polynomial_trait_methods() {
        let p = Poly2::<f64>::new([1.0, 2.0, 3.0]);

        assert_eq!(p.len(), 3);
        assert_eq!(p.coeffs(), &[1.0, 2.0, 3.0]);

        // Test out-of-bounds coefficient access
        assert_eq!(p.coeff(10), None);
    }

    #[test]
    fn test_polynomial_with_floats() {
        // Test with floating point coefficients
        let p1 = Poly1::<f64>::new([1.5, 2.5]);
        let p2 = Poly1::<f64>::new([0.5, 1.5]);
        let result = p1 + p2;

        assert!((result.coeff(0).unwrap() - 2.0).abs() < f64::EPSILON);
        assert!((result.coeff(1).unwrap() - 4.0).abs() < f64::EPSILON);

        // Test evaluation
        assert!((result.eval_horner(2.0) - 10.0).abs() < f64::EPSILON); // 2 + 4*2 = 10
    }
}
