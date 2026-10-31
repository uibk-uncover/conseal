
use std::cell::RefCell;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;


// ---------- internal helpers, NOT exposed to Python ----------

/// Digit-wise arithmetic on vectors of GF(q)^p, coded as base-q integers.
trait Field {
    fn add(&self, a: u64, b: u64) -> u64;
    fn neg(&self, a: u64) -> u64;
    fn mul(&self, c: u64, a: u64) -> u64;
}

/// GF(2)^p, digit-wise arithmetic is XOR.
struct Binary;

impl Field for Binary {
    fn add(&self, a: u64, b: u64) -> u64 { a ^ b }
    fn neg(&self, a: u64) -> u64 { a }
    fn mul(&self, c: u64, a: u64) -> u64 { if c & 1 == 1 { a } else { 0 } }
}

/// Lookup tables for GF(q), over limbs of L digits, Q = q^L <= 1024 values.
struct Tables {
    q: u64,
    digits_per_limb: u32,
    limb: u64,
    add: Vec<u64>,
    neg: Vec<u64>,
    mul: Vec<u64>,
}

impl Tables {
    fn new(q: u64) -> Self {
        // largest L with q^L <= 1024, at least 1
        let mut digits_per_limb = 1;
        while q.pow(digits_per_limb + 1) <= 1024 {
            digits_per_limb += 1;
        }
        let limb = q.pow(digits_per_limb);
        let digits = |x: u64| -> Vec<u64> {
            (0..digits_per_limb).map(|k| (x / q.pow(k)) % q).collect()
        };
        let compose = |d: &[u64]| -> u64 {
            d.iter().enumerate().map(|(k, &v)| v * q.pow(k as u32)).sum()
        };
        let all: Vec<Vec<u64>> = (0..limb).map(digits).collect();
        let mut add = vec![0u64; (limb * limb) as usize];
        for (i, di) in all.iter().enumerate() {
            for (j, dj) in all.iter().enumerate() {
                let d: Vec<u64> = di.iter().zip(dj).map(|(a, b)| (a + b) % q).collect();
                add[i * limb as usize + j] = compose(&d);
            }
        }
        let neg = all.iter()
            .map(|d| compose(&d.iter().map(|a| (q - a) % q).collect::<Vec<_>>()))
            .collect();
        let mut mul = vec![0u64; (q * limb) as usize];
        for c in 0..q {
            for (i, d) in all.iter().enumerate() {
                mul[(c * limb) as usize + i] = compose(&d.iter().map(|a| (c * a) % q).collect::<Vec<_>>());
            }
        }
        Tables { q, digits_per_limb, limb, add, neg, mul }
    }
}

/// GF(q)^p for prime q, by limb-wise table lookups.
struct Prime<'a> {
    tables: &'a Tables,
    num_limbs: u32,
}

impl Prime<'_> {
    fn limbwise(&self, mut a: u64, mut b: u64, f: impl Fn(usize, usize) -> u64) -> u64 {
        let q = self.tables.limb;
        let (mut result, mut base) = (0u64, 1u64);
        for _ in 0..self.num_limbs {
            result += f((a % q) as usize, (b % q) as usize) * base;
            a /= q;
            b /= q;
            base *= q;
        }
        result
    }
}

impl Field for Prime<'_> {
    fn add(&self, a: u64, b: u64) -> u64 {
        let limb = self.tables.limb as usize;
        self.limbwise(a, b, |x, y| self.tables.add[x * limb + y])
    }
    fn neg(&self, a: u64) -> u64 {
        self.limbwise(a, 0, |x, _| self.tables.neg[x])
    }
    fn mul(&self, c: u64, a: u64) -> u64 {
        let offset = (c * self.tables.limb) as usize;
        self.limbwise(a, 0, |x, _| self.tables.mul[offset + x])
    }
}

/// Workspace over all syndromes, reused across calls.
///
/// level is i + 1 for syndrome in C_i, 0 if unseen, and 1 for zero syndrome,
/// parent, column and coef give the syndrome one column less,
/// that column and its coefficient.
struct Workspace {
    level: Vec<u8>,
    parent: Vec<u32>,
    column: Vec<u32>,
    coef: Vec<u8>,
}

impl Workspace {
    fn ensure(&mut self, size: usize) {
        if self.level.len() < size {
            self.level.resize(size, 0);
            self.parent.resize(size, 0);
            self.column.resize(size, 0);
            self.coef.resize(size, 0);
        }
        self.level[0] = 1;
    }
}

thread_local! {
    static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace {
        level: Vec::new(), parent: Vec::new(), column: Vec::new(), coef: Vec::new(),
    });
    static TABLES: RefCell<Option<Tables>> = const { RefCell::new(None) };
}

/// Columns and coefficients of syndromes vs, summed, as solution of length n.
fn leader(ws: &Workspace, q: u64, vs: &[u64], n: usize) -> Vec<u8> {
    let mut v = vec![0u64; n];
    for &u in vs {
        let mut u = u as usize;
        while u != 0 {
            let j = ws.column[u] as usize;
            v[j] = (v[j] + ws.coef[u] as u64) % q;
            u = ws.parent[u] as usize;
        }
    }
    v.into_iter().map(|a| a as u8).collect()
}

/// Finds the fewest columns, with coefficients, adding up to s.
///
/// Meet-in-the-middle search, Algorithm 1 in Fridrich et al.,
/// Wet Paper Codes with Improved Embedding Efficiency.
/// Same order of sets and tie-breaking as the Python implementation,
/// so the solutions are identical.
fn coset_leader<F: Field>(
    field: &F,
    q: u64,
    columns: &[u64],
    s: u64,
    ws: &mut Workspace,
) -> Option<Vec<u8>> {
    let n = columns.len();
    if s == 0 {
        return Some(vec![0u8; n]);
    }

    // C_1, all multiples of the columns, sorted, first occurrence of each
    let mut multiples: Vec<(u64, usize)> = Vec::with_capacity(n * (q as usize - 1));
    for (j, &c) in columns.iter().enumerate() {
        for a in 1..q {
            multiples.push((field.mul(a, c), j * (q as usize - 1) + (a as usize - 1)));
        }
    }
    multiples.sort_unstable();
    multiples.dedup_by_key(|m| m.0);
    multiples.retain(|m| m.0 != 0);
    let values: Vec<u64> = multiples.iter().map(|m| m.0).collect();
    let columns1: Vec<u32> = multiples.iter().map(|m| (m.1 / (q as usize - 1)) as u32).collect();
    let coefs1: Vec<u8> = multiples.iter().map(|m| (m.1 % (q as usize - 1) + 1) as u8).collect();
    if values.is_empty() {
        return None;
    }
    for (i, &v) in values.iter().enumerate() {
        let v = v as usize;
        ws.level[v] = 2;
        ws.parent[v] = 0;
        ws.column[v] = columns1[i];
        ws.coef[v] = coefs1[i];
    }
    let mut sets: Vec<Vec<u64>> = vec![vec![0], values.clone()];

    let result = 'search: {
        // single column
        if ws.level[s as usize] == 2 {
            break 'search Some(leader(ws, q, &[s], n));
        }

        let (mut l, mut r) = (1usize, 1usize);
        loop {
            // is there x in C_l, s.t. s - x in C_r
            for &x in &sets[l] {
                let t = field.add(s, field.neg(x));
                if ws.level[t as usize] as usize == r + 1 {
                    break 'search Some(leader(ws, q, &[x, t], n));
                }
            }

            // next pair (l, r), creating C_r if needed
            if l < r {
                l += 1;
                continue;
            }
            r += 1;
            if r < sets.len() {
                continue;
            }

            // C_r = C_{r-1} + C_1, without syndromes of fewer columns,
            // in chunks of the frontier, each sorted, as in Python
            let frontier = sets.last().unwrap().clone();
            let chunk = std::cmp::max(1, (1usize << 22) / values.len());
            let mut created: Vec<u64> = Vec::new();
            for part in frontier.chunks(chunk) {
                let mut new: Vec<u64> = Vec::new();
                for &u in part {
                    for (i, &c) in values.iter().enumerate() {
                        let w = field.add(u, c);
                        let wi = w as usize;
                        if ws.level[wi] == 0 {
                            ws.level[wi] = (r + 1) as u8;
                            ws.parent[wi] = u as u32;
                            ws.column[wi] = columns1[i];
                            ws.coef[wi] = coefs1[i];
                            new.push(w);
                        }
                    }
                }
                new.sort_unstable();
                created.extend(new);
            }
            if created.is_empty() {
                sets.push(created);
                break 'search None;
            }
            sets.push(created);
        }
    };

    // reset the workspace
    for set in &sets[1..] {
        for &v in set {
            ws.level[v as usize] = 0;
        }
    }
    result
}


/// Solves Dv = s over GF(q) with the fewest nonzeros in v.
///
/// Columns of D and s are base-q integers of m digits.
/// Returns None, if there is no solution.
pub(crate) fn solve(columns: &[u64], s: u64, q: u64, m: usize) -> Option<Vec<u8>> {
    WORKSPACE.with(|ws| {
        let mut ws = ws.borrow_mut();
        ws.ensure(q.pow(m as u32) as usize);
        if q == 2 {
            coset_leader(&Binary, q, columns, s, &mut ws)
        } else {
            TABLES.with(|tables| {
                let mut tables = tables.borrow_mut();
                if tables.as_ref().is_none_or(|t| t.q != q) {
                    *tables = Some(Tables::new(q));
                }
                let tables = tables.as_ref().unwrap();
                let field = Prime { tables, num_limbs: (m as u32).div_ceil(tables.digits_per_limb) };
                coset_leader(&field, q, columns, s, &mut ws)
            })
        }
    })
}


// ---------- Python interface ----------

/// Solves Dv = z over GF(q) with the fewest nonzeros in v.
///
/// Block Minimal method with meet-in-the-middle search.
///
/// Parameters
/// ----------
/// d : np.ndarray
///     matrix of shape [m, n] over GF(q), uint8, q^m <= 2^22
/// z : np.ndarray
///     right-hand side of shape [m] over GF(q), uint8
/// q : int
///     prime field size
///
/// Returns
/// -------
/// np.ndarray
///     solution v of shape [n], uint8
#[pyfunction]
#[pyo3(signature = (d, z, q))]
fn lt_solve<'py>(
    py: Python<'py>,
    d: PyReadonlyArray2<'py, u8>,
    z: PyReadonlyArray1<'py, u8>,
    q: u64,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let d = d.as_array();
    let z = z.as_array();
    let (m, n) = d.dim();

    // columns and right-hand side as base-q integers
    let weights: Vec<u64> = (0..m as u32).map(|k| q.pow(k)).collect();
    let columns: Vec<u64> = (0..n)
        .map(|j| (0..m).map(|k| d[[k, j]] as u64 * weights[k]).sum())
        .collect();
    let s: u64 = (0..m).map(|k| z[k] as u64 * weights[k]).sum();

    let v = solve(&columns, s, q, m);
    match v {
        Some(v) => Ok(v.into_pyarray(py)),
        None => Err(PyValueError::new_err("system has no solution")),
    }
}

pub fn init_lt_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lt_solve, m)?)?;
    Ok(())
}
