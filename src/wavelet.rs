// Shared helpers of the wavelet-based cost functions (WOW, S-UNIWARD, J-UNIWARD),
// NOT exposed to Python.

/// Daubechies 8 wavelet filters, as separable pairs (vertical, horizontal).
///
/// The 2D filters are outer products: LH = l h^T, HL = h l^T, HH = h h^T.
pub fn daubechies8() -> [(Vec<f64>, Vec<f64>); 3] {
    let hpdf: [f64; 16] = [
        -0.0544158422,  0.3128715909, -0.6756307363,  0.5853546837,
         0.0158291053, -0.2840155430, -0.0004724846,  0.1287474266,
         0.0173693010, -0.0440882539, -0.0139810279,  0.0087460940,
         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768
    ];

    // build lpdf
    let mut lpdf = [0f64; 16];
    for i in 0..16 {
        lpdf[i] = ((-1f64).powi(i as i32)) * hpdf[15 - i];
    }

    [
        (lpdf.to_vec(), hpdf.to_vec()),
        (hpdf.to_vec(), lpdf.to_vec()),
        (hpdf.to_vec(), hpdf.to_vec()),
    ]
}

/// Mirror index into [0, n), edge included (SciPy boundary='symm').
pub fn reflect_index(i: isize, n: isize) -> usize {
    (if i < 0 { -i - 1 } else if i < n { i } else { 2 * n - i - 1 }) as usize
}

/// Symmetric padding of a row-major image of shape [h, w].
pub fn pad_symmetric(input: &[f64], h: usize, w: usize, pad: usize) -> (Vec<f64>, usize, usize) {
    let (hp, wp) = (h + 2 * pad, w + 2 * pad);
    let cols: Vec<usize> = (0..wp)
        .map(|j| reflect_index(j as isize - pad as isize, w as isize))
        .collect();
    let mut output = vec![0f64; hp * wp];
    for i in 0..hp {
        let ii = reflect_index(i as isize - pad as isize, h as isize);
        let src = &input[ii * w..(ii + 1) * w];
        for (dst, &jj) in output[i * wp..(i + 1) * wp].iter_mut().zip(&cols) {
            *dst = src[jj];
        }
    }
    (output, hp, wp)
}

/// 2D convolution with the separable kernel a b^T, mode='valid'.
///
/// out[i, j] = sum_{u, v} input[i + u, j + v] * a[ka-1-u] * b[kb-1-v]
pub fn convolve_separable(input: &[f64], h: usize, w: usize, a: &[f64], b: &[f64]) -> (Vec<f64>, usize, usize) {
    let (ka, kb) = (a.len(), b.len());
    let (ho, wo) = (h - ka + 1, w - kb + 1);

    // vertical pass, accumulating whole rows
    let mut tmp = vec![0f64; ho * w];
    for i in 0..ho {
        let dst = &mut tmp[i * w..(i + 1) * w];
        for u in 0..ka {
            let c = a[ka - 1 - u];
            let src = &input[(i + u) * w..(i + u + 1) * w];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += c * s;
            }
        }
    }

    // horizontal pass, flipped kernel
    let b_rev: Vec<f64> = b.iter().rev().copied().collect();
    let mut output = vec![0f64; ho * wo];
    for i in 0..ho {
        let src = &tmp[i * w..(i + 1) * w];
        for (j, d) in output[i * wo..(i + 1) * wo].iter_mut().enumerate() {
            *d = src[j..j + kb].iter().zip(&b_rev).map(|(s, c)| s * c).sum();
        }
    }
    (output, ho, wo)
}

/// 2D correlation with the separable kernel a b^T, mode='valid'.
///
/// out[i, j] = sum_{u, v} input[i + u, j + v] * a[u] * b[v]
pub fn correlate_separable(input: &[f64], h: usize, w: usize, a: &[f64], b: &[f64]) -> (Vec<f64>, usize, usize) {
    let a_rev: Vec<f64> = a.iter().rev().copied().collect();
    let b_rev: Vec<f64> = b.iter().rev().copied().collect();
    convolve_separable(input, h, w, &a_rev, &b_rev)
}

/// Crops a row-major image of shape [h, w] to [ho, wo], starting at (top, left).
pub fn crop(input: &[f64], w: usize, top: usize, left: usize, ho: usize, wo: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(ho * wo);
    for i in top..top + ho {
        out.extend_from_slice(&input[i * w + left..i * w + left + wo]);
    }
    out
}
