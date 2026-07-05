#[cfg(test)]
mod tests_duals {
    use crate::{ComparisonOp, LinearExpr, OptimizationDirection, Problem};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn duals_minimize() {
        // min 2x + 3y  s.t.  x + y ≥ 4,  x ≤ 3  (x, y ≥ 0) → x = 3, y = 1, obj 9.
        // Raising the covering rhs by 1 costs one more y (+3); raising the x cap
        // by 1 swaps a y for a cheaper x (−1).
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(2.0, (0.0, f64::INFINITY));
        let y = p.add_var(3.0, (0.0, f64::INFINITY));
        p.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 4.0);
        p.add_constraint([(x, 1.0)], ComparisonOp::Le, 3.0);
        let mut sol = p.solve().unwrap();
        assert_close(sol.objective(), 9.0);
        let duals = sol.dual_values().unwrap();
        assert_eq!(duals.len(), 2);
        assert_close(duals[0], 3.0);
        assert_close(duals[1], -1.0);
    }

    #[test]
    fn duals_maximize_and_nonbinding() {
        // max 2x + 3y  s.t.  x + y ≤ 4,  x ≤ 3 → x = 0, y = 4, obj 12.
        // The packing row prices at 3 (one more unit admits one more y);
        // the x cap is slack and prices at 0.
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(2.0, (0.0, f64::INFINITY));
        let y = p.add_var(3.0, (0.0, f64::INFINITY));
        p.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
        p.add_constraint([(x, 1.0)], ComparisonOp::Le, 3.0);
        let mut sol = p.solve().unwrap();
        assert_close(sol.objective(), 12.0);
        let duals = sol.dual_values().unwrap();
        assert_close(duals[0], 3.0);
        assert_close(duals[1], 0.0);
    }

    #[test]
    fn duals_equality_row() {
        // min x + 2y  s.t.  x + y = 5  (x, y ≥ 0) → x = 5, y = 0, obj 5.
        // The equality row prices at the cheaper variable's cost.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, f64::INFINITY));
        let y = p.add_var(2.0, (0.0, f64::INFINITY));
        p.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Eq, 5.0);
        let mut sol = p.solve().unwrap();
        assert_close(sol.objective(), 5.0);
        let duals = sol.dual_values().unwrap();
        assert_close(duals[0], 1.0);
    }

    #[test]
    fn duals_skip_tautological_rows() {
        // A tautological (empty left-hand side) constraint occupies no solver row
        // but must still report a zero dual at its caller-order position.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, f64::INFINITY));
        p.add_constraint([(x, 1.0)], ComparisonOp::Ge, 2.0);
        p.add_constraint(LinearExpr::empty(), ComparisonOp::Le, 5.0); // 0 ≤ 5: dropped
        p.add_constraint([(x, 1.0)], ComparisonOp::Le, 7.0);
        let mut sol = p.solve().unwrap();
        assert_close(sol.objective(), 2.0);
        let duals = sol.dual_values().unwrap();
        assert_eq!(duals.len(), 3);
        assert_close(duals[0], 1.0);
        assert_close(duals[1], 0.0);
        assert_close(duals[2], 0.0);
    }

    #[test]
    fn duals_after_incremental_add_constraint() {
        // min 2x + 3y  s.t.  x + y ≥ 4 → x = 4, obj 8, dual (2).
        // Then add x ≤ 3 through Solution::add_constraint (plus a tautological
        // row to exercise the caller-order map): x = 3, y = 1, obj 9,
        // duals (3, 0, −1) — same optimum as `duals_minimize`.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(2.0, (0.0, f64::INFINITY));
        let y = p.add_var(3.0, (0.0, f64::INFINITY));
        p.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 4.0);
        let mut sol = p.solve().unwrap();
        assert_close(sol.objective(), 8.0);
        let duals = sol.dual_values().unwrap();
        assert_eq!(duals.len(), 1);
        assert_close(duals[0], 2.0);

        let sol = sol
            .add_constraint(LinearExpr::empty(), ComparisonOp::Ge, -1.0)
            .unwrap();
        let mut sol = sol.add_constraint([(x, 1.0)], ComparisonOp::Le, 3.0).unwrap();
        assert_close(sol.objective(), 9.0);
        let duals = sol.dual_values().unwrap();
        assert_eq!(duals.len(), 3);
        assert_close(duals[0], 3.0);
        assert_close(duals[1], 0.0);
        assert_close(duals[2], -1.0);
    }

    #[test]
    fn duals_price_added_columns() {
        // The certifying property callers rely on: for a minimization problem, a
        // candidate column (a, c) improves the optimum iff its reduced cost
        // c − yᵀa is negative.
        //
        // min 2x + 3y  s.t.  x + y ≥ 4,  x ≤ 3 → obj 9, y = (3, −1).
        // Column z with a = (1, 0), c = 4: reduced cost 4 − 3 = 1 ≥ 0 → adding z
        // must leave the optimum at 9. Column w with a = (1, 0), c = 1:
        // reduced cost 1 − 3 = −2 < 0 → the optimum must drop.
        let build = |extra: Option<f64>| {
            let mut p = Problem::new(OptimizationDirection::Minimize);
            let x = p.add_var(2.0, (0.0, f64::INFINITY));
            let y = p.add_var(3.0, (0.0, f64::INFINITY));
            let e = extra.map(|c| p.add_var(c, (0.0, f64::INFINITY)));
            let mut cover = vec![(x, 1.0), (y, 1.0)];
            if let Some(e) = e {
                cover.push((e, 1.0));
            }
            p.add_constraint(cover, ComparisonOp::Ge, 4.0);
            p.add_constraint([(x, 1.0)], ComparisonOp::Le, 3.0);
            p.solve().unwrap()
        };
        let mut base = build(None);
        let duals = base.dual_values().unwrap();
        let price = |c: f64| c - duals[0]; // a = (1, 0)
        assert!(price(4.0) >= 0.0);
        assert_close(build(Some(4.0)).objective(), base.objective());
        assert!(price(1.0) < 0.0);
        assert!(build(Some(1.0)).objective() < base.objective() - 1e-9);
    }
}
