//! Warm incremental updates (`Solution::update`/`apply`/`add_var`):
//! correctness against cold rebuilds, batching, and re-solve cost.

use std::time::Instant;

use microlp::{ComparisonOp, OptimizationDirection, Problem, Variable};

#[test]
fn add_var_matches_cold_solve_small() {
    // min x + 2y  s.t.  x + y ≥ 2 → x = 2, obj 2.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(1.0, (0.0, f64::INFINITY));
    let y = p.add_var(2.0, (0.0, f64::INFINITY));
    p.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 2.0);
    let sol = p.solve().unwrap();
    assert!((sol.objective() - 2.0).abs() < 1e-9);

    // Non-improving column: optimum unchanged, var stays 0.
    let (sol, w) = sol.add_var(5.0, (0.0, f64::INFINITY), &[(0, 1.0)]).unwrap();
    assert!((sol.objective() - 2.0).abs() < 1e-9);
    assert!(sol[w].abs() < 1e-9);

    // Improving column: takes over.
    let (sol, z) = sol.add_var(0.5, (0.0, f64::INFINITY), &[(0, 1.0)]).unwrap();
    assert!((sol.objective() - 1.0).abs() < 1e-9);
    assert!((sol[z] - 2.0).abs() < 1e-9);

    // A warm ROW after the warm columns still composes.
    let sol = sol.add_constraint([(z, 1.0)], ComparisonOp::Le, 0.5).unwrap();
    assert!((sol.objective() - 1.75).abs() < 1e-9);
}

#[test]
fn batch_with_new_rows_referencing_new_vars() {
    // Flow-LP-shaped delta: a new weight column plus new capacity rows
    // that reference BOTH the new flow column and the new weight.
    // Base: min 3w1 s.t. f1 ≤ w1 (as f1 − w1 ≤ 0), f1 ≥ 1.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let w1 = p.add_var(3.0, (0.0, f64::INFINITY));
    let f1 = p.add_var(0.0, (0.0, f64::INFINITY));
    p.add_constraint([(f1, 1.0), (w1, -1.0)], ComparisonOp::Le, 0.0);
    p.add_constraint([(f1, 1.0)], ComparisonOp::Ge, 1.0);
    let sol = p.solve().unwrap();
    assert!((sol.objective() - 3.0).abs() < 1e-9);

    // Delta: a cheaper parallel route — weight w2 (cost 1), flow f2,
    // capacity f2 − w2 ≤ 0, and the demand row loosens to f1 + f2 ≥ 1
    // (modeled here as a fresh demand row on f2 replacing pressure on
    // f1 via an Le cap on f1).
    let mut u = sol.update();
    let w2 = u.add_var(1.0, (0.0, f64::INFINITY), &[]);
    let f2 = u.add_var(0.0, (0.0, f64::INFINITY), &[(1, 1.0)]); // joins the ≥1 demand row
    u.add_constraint([(f2, 1.0), (w2, -1.0)], ComparisonOp::Le, 0.0);
    let sol = sol.apply(u).unwrap();
    // Cheapest now: route the demand through f2/w2 at cost 1.
    assert!((sol.objective() - 1.0).abs() < 1e-9, "obj {}", sol.objective());
    assert!((sol[f2] - 1.0).abs() < 1e-9);
    assert!(sol[w1].abs() < 1e-9);

    // Cold rebuild agrees.
    let mut p2 = Problem::new(OptimizationDirection::Minimize);
    let cw1 = p2.add_var(3.0, (0.0, f64::INFINITY));
    let cf1 = p2.add_var(0.0, (0.0, f64::INFINITY));
    let cw2 = p2.add_var(1.0, (0.0, f64::INFINITY));
    let cf2 = p2.add_var(0.0, (0.0, f64::INFINITY));
    p2.add_constraint([(cf1, 1.0), (cw1, -1.0)], ComparisonOp::Le, 0.0);
    p2.add_constraint([(cf1, 1.0), (cf2, 1.0)], ComparisonOp::Ge, 1.0);
    p2.add_constraint([(cf2, 1.0), (cw2, -1.0)], ComparisonOp::Le, 0.0);
    let cold = p2.solve().unwrap();
    assert!((cold.objective() - sol.objective()).abs() < 1e-9);
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Base problem + batch as plain data, so warm and cold builds share one source.
struct Spec {
    // (obj, min, max) — bounds capped so nothing is unbounded.
    vars: Vec<(f64, f64, f64)>,
    // (terms over var idx, op, rhs) — Le with generous rhs / Ge with small
    // rhs over nonnegative coeffs: always feasible under (0, 10) bounds.
    rows: Vec<(Vec<(usize, f64)>, ComparisonOp, f64)>,
    batch_vars: Vec<(f64, f64, f64, Vec<(usize, f64)>)>, // + coeffs on existing rows
    batch_rows: Vec<(Vec<(usize, f64)>, ComparisonOp, f64)>, // over old + new var idx
}

fn random_spec(seed: u64) -> Spec {
    let mut s = seed;
    let nv = 4 + (lcg(&mut s) % 12) as usize;
    let vars: Vec<(f64, f64, f64)> = (0..nv)
        .map(|_| (0.5 + (lcg(&mut s) % 8) as f64 * 0.5, 0.0, 10.0))
        .collect();
    let nr = 3 + (lcg(&mut s) % 8) as usize;
    let mut rows = Vec::new();
    for r in 0..nr {
        let nt = 2 + (lcg(&mut s) % 3) as usize;
        let mut terms: Vec<(usize, f64)> = Vec::new();
        for _ in 0..nt {
            let v = (lcg(&mut s) % nv as u64) as usize;
            if !terms.iter().any(|&(tv, _)| tv == v) {
                terms.push((v, 0.5 + (lcg(&mut s) % 5) as f64 * 0.5));
            }
        }
        let ge = r % 2 == 0;
        if ge {
            rows.push((terms, ComparisonOp::Ge, 0.5 + (lcg(&mut s) % 3) as f64));
        } else {
            rows.push((terms, ComparisonOp::Le, 20.0 + (lcg(&mut s) % 20) as f64));
        }
    }
    let bk = (lcg(&mut s) % 3) as usize;
    let mut batch_vars = Vec::new();
    for _ in 0..bk {
        let mut coeffs: Vec<(usize, f64)> = Vec::new();
        for _ in 0..(lcg(&mut s) % 3) {
            let r = (lcg(&mut s) % nr as u64) as usize;
            if !coeffs.iter().any(|&(cr, _)| cr == r) {
                coeffs.push((r, 0.5 + (lcg(&mut s) % 4) as f64 * 0.5));
            }
        }
        batch_vars.push((
            0.25 + (lcg(&mut s) % 8) as f64 * 0.25,
            0.0,
            10.0,
            coeffs,
        ));
    }
    let br = (lcg(&mut s) % 3) as usize;
    let total_v = nv + bk;
    let mut batch_rows = Vec::new();
    for i in 0..br {
        let nt = 2 + (lcg(&mut s) % 3) as usize;
        let mut terms: Vec<(usize, f64)> = Vec::new();
        for _ in 0..nt {
            let v = (lcg(&mut s) % total_v as u64) as usize;
            if !terms.iter().any(|&(tv, _)| tv == v) {
                terms.push((v, 0.5 + (lcg(&mut s) % 5) as f64 * 0.5));
            }
        }
        if i % 2 == 0 {
            batch_rows.push((terms, ComparisonOp::Ge, 0.5 + (lcg(&mut s) % 2) as f64));
        } else {
            batch_rows.push((terms, ComparisonOp::Le, 25.0 + (lcg(&mut s) % 10) as f64));
        }
    }
    Spec { vars, rows, batch_vars, batch_rows }
}

/// FUZZ GATE: for every seed, warm apply == cold rebuild within ε.
#[test]
fn warm_apply_matches_cold_rebuild_fuzz() {
    for seed in 0..300u64 {
        let spec = random_spec(seed);

        // Cold base.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let vs: Vec<Variable> = spec
            .vars
            .iter()
            .map(|&(c, lo, hi)| p.add_var(c, (lo, hi)))
            .collect();
        for (terms, op, rhs) in &spec.rows {
            let expr: Vec<(Variable, f64)> =
                terms.iter().map(|&(v, c)| (vs[v], c)).collect();
            p.add_constraint(expr, *op, *rhs);
        }
        let sol = p.solve().expect("base LP is feasible by construction");

        // Warm batch.
        let mut u = sol.update();
        let mut all: Vec<Variable> = vs.clone();
        for (c, lo, hi, coeffs) in &spec.batch_vars {
            all.push(u.add_var(*c, (*lo, *hi), coeffs));
        }
        for (terms, op, rhs) in &spec.batch_rows {
            let expr: Vec<(Variable, f64)> =
                terms.iter().map(|&(v, c)| (all[v], c)).collect();
            u.add_constraint(expr, *op, *rhs);
        }
        let warm = sol.apply(u).expect("batch keeps the LP feasible");

        // Cold rebuild of the combined problem: existing rows carry the
        // new vars' coefficients from the start.
        let mut p2 = Problem::new(OptimizationDirection::Minimize);
        let mut vs2: Vec<Variable> = spec
            .vars
            .iter()
            .map(|&(c, lo, hi)| p2.add_var(c, (lo, hi)))
            .collect();
        for (c, lo, hi, _) in &spec.batch_vars {
            vs2.push(p2.add_var(*c, (*lo, *hi)));
        }
        for (r, (terms, op, rhs)) in spec.rows.iter().enumerate() {
            let mut expr: Vec<(Variable, f64)> =
                terms.iter().map(|&(v, c)| (vs2[v], c)).collect();
            for (i, (_, _, _, coeffs)) in spec.batch_vars.iter().enumerate() {
                for &(cr, cc) in coeffs {
                    if cr == r {
                        expr.push((vs2[spec.vars.len() + i], cc));
                    }
                }
            }
            p2.add_constraint(expr, *op, *rhs);
        }
        for (terms, op, rhs) in &spec.batch_rows {
            let expr: Vec<(Variable, f64)> =
                terms.iter().map(|&(v, c)| (vs2[v], c)).collect();
            p2.add_constraint(expr, *op, *rhs);
        }
        let cold = p2.solve().expect("cold rebuild feasible");

        let (w, c) = (warm.objective(), cold.objective());
        assert!(
            (w - c).abs() <= 1e-6 + 1e-9 * c.abs(),
            "seed {seed}: warm {w} vs cold {c}"
        );
    }
}

/// SEQUENCE fuzz: several batches applied to ONE retained solution
/// (identity-border growth, pivots, further growth on top of etas —
/// the drain-loop shape), each step checked against a cold rebuild of
/// the accumulated problem.
#[test]
fn warm_sequences_match_cold_rebuild_fuzz() {
    for seed in 1000..1150u64 {
        let mut s = seed;
        // Accumulated problem as data.
        let base = random_spec(seed);
        let mut vars: Vec<(f64, f64, f64)> = base.vars.clone();
        let mut rows: Vec<(Vec<(usize, f64)>, ComparisonOp, f64)> = base.rows.clone();

        let mut p = Problem::new(OptimizationDirection::Minimize);
        let vs: Vec<Variable> = vars.iter().map(|&(c, lo, hi)| p.add_var(c, (lo, hi))).collect();
        for (terms, op, rhs) in &rows {
            let expr: Vec<(Variable, f64)> = terms.iter().map(|&(v, c)| (vs[v], c)).collect();
            p.add_constraint(expr, *op, *rhs);
        }
        let mut warm = p.solve().expect("base feasible");
        let mut handles = vs;

        for _round in 0..3 {
            // Random batch over the CURRENT accumulated problem.
            let nv = vars.len();
            let nr = rows.len();
            let mut u = warm.update();
            let bk = (lcg(&mut s) % 3) as usize;
            for _ in 0..bk {
                let mut coeffs: Vec<(usize, f64)> = Vec::new();
                for _ in 0..(lcg(&mut s) % 3) {
                    let r = (lcg(&mut s) % nr as u64) as usize;
                    if !coeffs.iter().any(|&(cr, _)| cr == r) {
                        coeffs.push((r, 0.5 + (lcg(&mut s) % 4) as f64 * 0.5));
                    }
                }
                let obj = 0.25 + (lcg(&mut s) % 8) as f64 * 0.25;
                handles.push(u.add_var(obj, (0.0, 10.0), &coeffs));
                vars.push((obj, 0.0, 10.0));
                // record coeffs into the accumulated rows for the cold build
                for &(r, c) in &coeffs {
                    rows[r].0.push((vars.len() - 1, c));
                }
            }
            let br = (lcg(&mut s) % 3) as usize;
            for i in 0..br {
                let total_v = vars.len();
                let nt = 2 + (lcg(&mut s) % 3) as usize;
                let mut terms: Vec<(usize, f64)> = Vec::new();
                for _ in 0..nt {
                    let v = (lcg(&mut s) % total_v as u64) as usize;
                    if !terms.iter().any(|&(tv, _)| tv == v) {
                        terms.push((v, 0.5 + (lcg(&mut s) % 5) as f64 * 0.5));
                    }
                }
                let (op, rhs) = if i % 2 == 0 {
                    (ComparisonOp::Ge, 0.5 + (lcg(&mut s) % 2) as f64)
                } else {
                    (ComparisonOp::Le, 25.0 + (lcg(&mut s) % 10) as f64)
                };
                let expr: Vec<(Variable, f64)> =
                    terms.iter().map(|&(v, c)| (handles[v], c)).collect();
                u.add_constraint(expr, op, rhs);
                rows.push((terms, op, rhs));
            }
            warm = warm.apply(u).expect("batch keeps the LP feasible");

            // Cold rebuild of the accumulated problem.
            let mut p2 = Problem::new(OptimizationDirection::Minimize);
            let vs2: Vec<Variable> =
                vars.iter().map(|&(c, lo, hi)| p2.add_var(c, (lo, hi))).collect();
            for (terms, op, rhs) in &rows {
                let expr: Vec<(Variable, f64)> =
                    terms.iter().map(|&(v, c)| (vs2[v], c)).collect();
                p2.add_constraint(expr, *op, *rhs);
            }
            let cold = p2.solve().expect("cold rebuild feasible");
            let (w, c) = (warm.objective(), cold.objective());
            assert!(
                (w - c).abs() <= 1e-6 + 1e-9 * c.abs(),
                "seed {seed}: warm {w} vs cold {c}"
            );
        }
    }
}

/// The C≠0 fallback: a new row referencing an old BASIC variable makes
/// the border block nonzero — the fast path must refuse and the
/// refactorizing fallback must still land on the cold optimum.
#[test]
fn c_nonzero_row_falls_back_and_matches_cold() {
    // min x + 2y  s.t.  x + y ≥ 2 → x = 2 (x is BASIC at the optimum).
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(1.0, (0.0, f64::INFINITY));
    let y = p.add_var(2.0, (0.0, f64::INFINITY));
    p.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 2.0);
    let sol = p.solve().unwrap();
    assert!((sol[x] - 2.0).abs() < 1e-9, "x is basic at 2");

    // New row caps the BASIC x at 1.5: C ≠ 0, and the row is violated
    // at the current point — the fallback must refactorize, restore
    // feasibility, and re-optimize.
    let mut u = sol.update();
    u.add_constraint([(x, 1.0)], ComparisonOp::Le, 1.5);
    let warm = sol.apply(u).unwrap();
    // Cold: x = 1.5, y = 0.5 → obj 1.5 + 1.0 = 2.5.
    assert!((warm.objective() - 2.5).abs() < 1e-9, "obj {}", warm.objective());
    assert!((warm[x] - 1.5).abs() < 1e-9);
    assert!((warm[y] - 0.5).abs() < 1e-9);
}

/// Transportation-shaped LP at roughly the flow-LP scale.
fn build_transportation(
    m: usize,
    n: usize,
    extra: &[(usize, usize, f64)],
) -> (Problem, Vec<(Variable, usize, usize)>) {
    let mut s = 12345u64;
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let mut arcs: Vec<(Variable, usize, usize)> = Vec::new();
    let mut arc_list: Vec<(usize, usize, f64)> = Vec::new();
    for j in 0..n {
        for _ in 0..3 {
            let i = (lcg(&mut s) % m as u64) as usize;
            arc_list.push((i, j, 1.0 + (lcg(&mut s) % 9) as f64));
        }
    }
    arc_list.extend_from_slice(extra);
    for &(i, j, cost) in &arc_list {
        let v = p.add_var(cost, (0.0, f64::INFINITY));
        arcs.push((v, i, j));
    }
    for i in 0..m {
        let terms: Vec<(Variable, f64)> = arcs
            .iter()
            .filter(|&&(_, ai, _)| ai == i)
            .map(|&(v, _, _)| (v, 1.0))
            .collect();
        p.add_constraint(terms, ComparisonOp::Le, 3.0 * n as f64 / m as f64);
    }
    for j in 0..n {
        let terms: Vec<(Variable, f64)> = arcs
            .iter()
            .filter(|&&(_, _, aj)| aj == j)
            .map(|&(v, _, _)| (v, 1.0))
            .collect();
        p.add_constraint(terms, ComparisonOp::Ge, 1.0);
    }
    (p, arcs)
}

/// Steady-state warm-apply cost: many sequential batches on one
/// retained solution (the incremental caller's real shape), reported
/// as an average.
#[test]
fn warm_apply_steady_state_cost() {
    let m = 60usize;
    let n = 950usize;
    let (p, _) = build_transportation(m, n, &[]);
    let mut sol = p.solve().unwrap();

    // Columns-only steady state (non-improving: zero pivots).
    let t0 = Instant::now();
    let rounds_cols = 50;
    for i in 0..rounds_cols {
        let mut u = sol.update();
        u.add_var(50.0 + i as f64, (0.0, f64::INFINITY), &[(i % m, 1.0), (m + i % n, 1.0)]);
        sol = sol.apply(u).unwrap();
    }
    let cols_us = t0.elapsed().as_secs_f64() * 1e6 / rounds_cols as f64;

    // Mixed steady state: a column + a row per batch (C=0: the new row
    // references only the new column).
    let t1 = Instant::now();
    let rounds_mixed = 50;
    for i in 0..rounds_mixed {
        let mut u = sol.update();
        let v = u.add_var(60.0 + i as f64, (0.0, f64::INFINITY), &[(i % m, 1.0)]);
        u.add_constraint([(v, 1.0)], ComparisonOp::Le, 1.0);
        sol = sol.apply(u).unwrap();
    }
    let mixed_us = t1.elapsed().as_secs_f64() * 1e6 / rounds_mixed as f64;

    // An improving column at the end still lands on the right optimum.
    let (sol, good) = sol.add_var(0.01, (0.0, f64::INFINITY), &[(0, 1.0), (m + 3, 1.0)]).unwrap();
    assert!(sol[good] > 0.5);

    println!("steady-state warm apply: cols-only {cols_us:.1}µs | mixed (col+row) {mixed_us:.1}µs");
}

#[test]
fn warm_batch_vs_cold_at_scale() {
    let m = 60usize;
    let n = 950usize;

    let t0 = Instant::now();
    let (p, _) = build_transportation(m, n, &[]);
    let sol = p.solve().unwrap();
    let cold_ms = t0.elapsed().as_secs_f64() * 1e3;

    // A flow-fact-shaped batch: one improving arc + one dead arc + a
    // fresh Le row over the improving arc — all in ONE warm apply.
    let t1 = Instant::now();
    let mut u = sol.update();
    let good = u.add_var(0.01, (0.0, f64::INFINITY), &[(0, 1.0), (m + 5, 1.0)]);
    let dead = u.add_var(100.0, (0.0, f64::INFINITY), &[(1, 1.0), (m + 7, 1.0)]);
    u.add_constraint([(good, 1.0)], ComparisonOp::Le, 2.0);
    let warm = sol.apply(u).unwrap();
    let warm_ms = t1.elapsed().as_secs_f64() * 1e3;
    assert!(warm[dead].abs() < 1e-9);

    // Second batch: columns only, priced dead — the zero-pivot,
    // zero-refactorization path.
    let t2 = Instant::now();
    let mut u2 = warm.update();
    u2.add_var(90.0, (0.0, f64::INFINITY), &[(2, 1.0), (m + 9, 1.0)]);
    let warm2 = warm.apply(u2).unwrap();
    let cols_only_ms = t2.elapsed().as_secs_f64() * 1e3;

    // Cold rebuild agrees.
    let t3 = Instant::now();
    let (p2, arcs2) = build_transportation(
        m,
        n,
        &[(0, 5, 0.01), (1, 7, 100.0), (2, 9, 90.0)],
    );
    let good_cold = arcs2[arcs2.len() - 3].0;
    let mut p2 = p2;
    p2.add_constraint([(good_cold, 1.0)], ComparisonOp::Le, 2.0);
    let cold2 = p2.solve().unwrap();
    let cold2_ms = t3.elapsed().as_secs_f64() * 1e3;
    assert!(
        (cold2.objective() - warm2.objective()).abs() <= 1e-6 + 1e-9 * cold2.objective().abs(),
        "warm {} vs cold {}",
        warm2.objective(),
        cold2.objective()
    );

    println!(
        "cold: {cold_ms:.2}ms | warm batch (2 cols + 1 row): {warm_ms:.3}ms | \
         warm cols-only (no refactorization): {cols_only_ms:.3}ms | cold rebuild: {cold2_ms:.2}ms"
    );
}
