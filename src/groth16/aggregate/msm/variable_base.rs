use ff::{Field, PrimeField, PrimeFieldRepr};
use groupy::{CurveAffine, CurveProjective};

use rayon::prelude::*;

pub fn multi_scalar_mul<G: CurveAffine>(
    bases: &[G],
    scalars: &[<G::Scalar as PrimeField>::Repr],
) -> G::Projective {
    let c = if scalars.len() < 32 {
        3
    } else {
        (scalars.len() as f64).ln().ceil() as usize + 2
    };

    let num_bits = <G::Scalar as PrimeField>::NUM_BITS as usize;
    let fr_one = G::Scalar::one().into_repr();

    let zero = G::Projective::zero();
    let window_starts: Vec<_> = (0..num_bits).step_by(c).collect();

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // in parallel process each such window.
    let window_sums: Vec<_> = window_starts
        .par_iter()
        .map(|w_start| {
            let mut res = zero;
            // We don't need the "zero" bucket, so we only have 2^c - 1 buckets
            let mut buckets = vec![zero; (1 << c) - 1];
            scalars
                .iter()
                .zip(bases.iter())
                .filter(|(s, _)| !s.is_zero())
                .for_each(|(&scalar, base)| {
                    if scalar == fr_one {
                        // We only process unit scalars once in the first window.
                        if *w_start == 0 {
                            res.add_assign_mixed(&base);
                        }
                    } else {
                        let mut scalar = scalar;

                        // We right-shift by w_start, thus getting rid of the lower bits.
                        scalar.shr(*w_start as u32);

                        // We mod the remaining bits by the window size.
                        let scalar = scalar.as_ref()[0] % (1 << c);

                        // If the scalar is non-zero, we update the corresponding
                        // bucket.
                        // (Recall that `buckets` doesn't have a zero bucket.)
                        if scalar != 0 {
                            buckets[(scalar - 1) as usize].add_assign_mixed(&base);
                        }
                    }
                });

            let mut running_sum = G::Projective::zero();
            for b in buckets.into_iter().rev() {
                running_sum.add_assign_mixed(&b.into_affine());
                res.add_assign(&running_sum);
            }

            res
        })
        .collect();

    // We store the sum for the lowest window.
    let mut res = *window_sums.first().unwrap();

    // We're traversing windows from high to low.
    res.add_assign(&window_sums[1..].into_par_iter().rev().copied().reduce(
        || zero,
        |mut total, sum_i| {
            total.add_assign(&sum_i);
            for _ in 0..c {
                total.double();
            }
            total
        },
    ));
    res
}
