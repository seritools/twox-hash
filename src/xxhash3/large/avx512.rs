use core::arch::x86_64::*;

use super::Vector;
use crate::xxhash3::primes::PRIME32_1;

#[derive(Copy, Clone)]
pub struct Impl(());

impl Impl {
    /// # Safety
    ///
    /// You must ensure that the CPU has the AVX512F feature
    #[inline]
    #[cfg(feature = "std")]
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Vector for Impl {
    #[inline]
    fn round_scramble(&self, acc: &mut [u64; 8], secret_end: &[u8; 64]) {
        // Safety: Type can only be constructed when AVX512F feature is present
        unsafe { round_scramble_avx512(acc, secret_end) }
    }

    #[inline]
    fn accumulate(&self, acc: &mut [u64; 8], stripe: &[u8; 64], secret: &[u8; 64]) {
        // Safety: Type can only be constructed when AVX512F feature is present
        unsafe { accumulate_avx512(acc, stripe, secret) }
    }
}

/// # Safety
///
/// You must ensure that the CPU has the AVX512F feature
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn round_scramble_avx512(acc: &mut [u64; 8], secret_end: &[u8; 64]) {
    let acc = acc.as_mut_ptr().cast::<__m512i>();
    let secret_end = secret_end.as_ptr().cast::<__m512i>();

    // Safety: The caller has ensured we have the AVX512F
    // feature. We load from and store to references so we know that
    // data is valid. We use unaligned loads / stores. Data
    // manipulation is otherwise done on intermediate values.
    unsafe {
        let prime_0 = _mm512_set1_epi32(PRIME32_1 as i32);

        // See [align-acc].
        let acc_0 = _mm512_loadu_si512(acc);
        let secret_0 = _mm512_loadu_si512(secret_end);

        // shifted[i] = acc[i] >> 47
        let shifted_0 = _mm512_srli_epi64::<47>(acc_0);

        // value[i] = acc[i] ^ shifted[i] ^ secret[i]
        let value_0 = _mm512_ternarylogic_epi64::<0x96>(acc_0, shifted_0, secret_0);

        // Build the product from its two halves, as the other
        // implementations do. `vpmullq` would do this in one
        // instruction but needs AVX512DQ, and where it is available it
        // measured about 10% slower overall (it is not a single-uop
        // instruction on the hardware tested).
        let value_hi_0 = _mm512_srli_epi64::<32>(value_0);
        let product_lo_0 = _mm512_mul_epu32(value_0, prime_0);
        let product_hi_0 = _mm512_mul_epu32(value_hi_0, prime_0);

        // acc[i] = value[i] * PRIME32_1
        let acc_0 = _mm512_add_epi64(product_lo_0, _mm512_slli_epi64::<32>(product_hi_0));

        _mm512_storeu_si512(acc, acc_0);
    }
}

/// # Safety
///
/// You must ensure that the CPU has the AVX512F feature
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_avx512(acc: &mut [u64; 8], stripe: &[u8; 64], secret: &[u8; 64]) {
    let acc = acc.as_mut_ptr().cast::<__m512i>();
    let stripe = stripe.as_ptr().cast::<__m512i>();
    let secret = secret.as_ptr().cast::<__m512i>();

    // Safety: The caller has ensured we have the AVX512F
    // feature. We load from and store to references so we know that
    // data is valid. We use unaligned loads / stores. Data
    // manipulation is otherwise done on intermediate values.
    unsafe {
        // See [align-acc].
        let acc_0 = _mm512_loadu_si512(acc);
        let stripe_0 = _mm512_loadu_si512(stripe);
        let secret_0 = _mm512_loadu_si512(secret);

        // let value[i] = stripe[i] ^ secret[i];
        let value_0 = _mm512_xor_si512(stripe_0, secret_0);

        // stripe_swap[i] = stripe[i ^ 1]
        let stripe_swap_0 = _mm512_shuffle_epi32::<0b01_00_11_10>(stripe_0);

        // acc[i] += stripe_swap[i]
        let acc_0 = _mm512_add_epi64(acc_0, stripe_swap_0);

        // value_shift[i] = value[i] >> 32
        let value_shift_0 = _mm512_srli_epi64::<32>(value_0);

        // product[i] = lower_32_bit(value[i]) * lower_32_bit(value_shift[i])
        let product_0 = _mm512_mul_epu32(value_0, value_shift_0);

        // acc[i] += product[i]
        let acc_0 = _mm512_add_epi64(acc_0, product_0);

        _mm512_storeu_si512(acc, acc_0);
    }
}
