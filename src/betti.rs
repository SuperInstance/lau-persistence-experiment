//! Betti curve tracking — how topology changes as learning progresses.

use crate::belief::StateHistory;
use crate::persistence::PersistenceDiagram;
use serde::{Deserialize, Serialize};

/// Betti numbers at a given filtration value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettiSnapshot {
    pub step: usize,
    pub filtration: f64,
    pub betti: Vec<usize>, // betti[0], betti[1], ...
}

/// A full Betti curve: sequence of Betti numbers over filtration values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettiCurve {
    pub snapshots: Vec<BettiSnapshot>,
    pub max_dim: usize,
}

impl BettiCurve {
    pub fn new(snapshots: Vec<BettiSnapshot>, max_dim: usize) -> Self {
        Self { snapshots, max_dim }
    }

    /// Get Betti-k at a specific step.
    pub fn betti_at_step(&self, step: usize, k: usize) -> Option<usize> {
        self.snapshots
            .iter()
            .find(|s| s.step == step)
            .and_then(|s| s.betti.get(k).copied())
    }

    /// Maximum Betti-0 value (most disconnected state).
    pub fn max_betti0(&self) -> usize {
        self.snapshots
            .iter()
            .map(|s| s.betti.first().copied().unwrap_or(0))
            .max()
            .unwrap_or(0)
    }

    /// Maximum Betti-1 value (most loops).
    pub fn max_betti1(&self) -> usize {
        self.snapshots
            .iter()
            .map(|s| s.betti.get(1).copied().unwrap_or(0))
            .max()
            .unwrap_or(0)
    }

    /// Area under the Betti-0 curve (total disconnectedness).
    pub fn betti0_area(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }
        let mut area = 0.0;
        for w in self.snapshots.windows(2) {
            let b0_a = w[0].betti.first().copied().unwrap_or(0) as f64;
            let b0_b = w[1].betti.first().copied().unwrap_or(0) as f64;
            let dt = (w[1].step as f64) - (w[0].step as f64);
            area += (b0_a + b0_b) / 2.0 * dt;
        }
        area
    }

    /// Area under the Betti-1 curve (total loopiness).
    pub fn betti1_area(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }
        let mut area = 0.0;
        for w in self.snapshots.windows(2) {
            let b1_a = w[0].betti.get(1).copied().unwrap_or(0) as f64;
            let b1_b = w[1].betti.get(1).copied().unwrap_or(0) as f64;
            let dt = (w[1].step as f64) - (w[0].step as f64);
            area += (b1_a + b1_b) / 2.0 * dt;
        }
        area
    }
}

/// Compute Betti numbers from a persistence diagram at a given filtration value.
pub fn betti_from_diagram(diagram: &PersistenceDiagram, filtration: f64, max_dim: usize) -> Vec<usize> {
    let mut betti = vec![0usize; max_dim + 1];
    for p in &diagram.pairs {
        if p.dim <= max_dim && p.birth <= filtration && (p.death > filtration || !p.death.is_finite()) {
            betti[p.dim] += 1;
        }
    }
    betti
}

/// Compute Betti curve over a learning trajectory using sliding windows.
pub fn compute_betti_curve(
    history: &StateHistory,
    window_size: usize,
    stride: usize,
    max_dim: usize,
    n_filtration_steps: usize,
) -> BettiCurve {
    let windows = crate::persistence::sliding_window_persistence(history, window_size, stride, max_dim);
    let mut snapshots = Vec::new();

    for (start, dg) in &windows {
        // Use a fixed filtration value based on the diagram
        let filtration = dg.max_persistence() * 0.5;
        let betti = betti_from_diagram(dg, filtration, max_dim);
        snapshots.push(BettiSnapshot {
            step: *start,
            filtration,
            betti,
        });
    }

    BettiCurve::new(snapshots, max_dim)
}

/// Compute Betti curve at multiple filtration values for a single diagram.
pub fn betti_curve_filtration(
    diagram: &PersistenceDiagram,
    max_dim: usize,
    n_steps: usize,
) -> BettiCurve {
    let max_f = diagram
        .pairs
        .iter()
        .filter(|p| p.is_finite())
        .map(|p| p.death)
        .fold(0.0_f64, f64::max);

    let mut snapshots = Vec::new();
    for i in 0..=n_steps {
        let f = max_f * (i as f64) / (n_steps as f64);
        let betti = betti_from_diagram(diagram, f, max_dim);
        snapshots.push(BettiSnapshot {
            step: i,
            filtration: f,
            betti,
        });
    }

    BettiCurve::new(snapshots, max_dim)
}

/// Compute Euler characteristic curve (alternating sum of Betti numbers).
pub fn euler_characteristic_curve(curve: &BettiCurve) -> Vec<(usize, i64)> {
    curve
        .snapshots
        .iter()
        .map(|s| {
            let euler: i64 = s
                .betti
                .iter()
                .enumerate()
                .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
                .sum();
            (s.step, euler)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::*;
    use crate::persistence::*;

    #[test]
    fn test_betti_from_empty_diagram() {
        let dg = PersistenceDiagram::empty();
        let betti = betti_from_diagram(&dg, 1.0, 2);
        assert_eq!(betti, vec![0, 0, 0]);
    }

    #[test]
    fn test_betti_single_component() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, f64::INFINITY, 0),
            ],
            1,
        );
        let betti = betti_from_diagram(&dg, 0.5, 1);
        assert_eq!(betti[0], 1);
    }

    #[test]
    fn test_betti_two_components() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0), // merged at f=1
                PersistencePair::new(0.0, f64::INFINITY, 0),
            ],
            2,
        );
        // At f=0.5, second component hasn't merged yet → 2 components
        let betti = betti_from_diagram(&dg, 0.5, 1);
        assert_eq!(betti[0], 2);
        // At f=1.5, merged → 1 component
        let betti = betti_from_diagram(&dg, 1.5, 1);
        assert_eq!(betti[0], 1);
    }

    #[test]
    fn test_betti_with_loop() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, f64::INFINITY, 0),
                PersistencePair::new(0.5, 2.0, 1),
            ],
            4,
        );
        let betti = betti_from_diagram(&dg, 1.0, 1);
        assert_eq!(betti[0], 1);
        assert_eq!(betti[1], 1);
    }

    #[test]
    fn test_betti_curve_basic() {
        let t = synthetic_trajectory(2, 40, 0.1, 0.01, 42);
        let curve = compute_betti_curve(&t, 10, 5, 1, 5);
        assert!(curve.snapshots.len() > 0);
    }

    #[test]
    fn test_betti_curve_filtration() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0, 0.0], 0),
            BeliefState::new(vec![1.0, 0.0], 1),
            BeliefState::new(vec![0.0, 1.0], 2),
        ]);
        let dg = compute_persistence(&h, 1);
        let curve = betti_curve_filtration(&dg, 1, 10);
        assert_eq!(curve.snapshots.len(), 11);
    }

    #[test]
    fn test_betti_curve_max_betti0() {
        let curve = BettiCurve::new(
            vec![
                BettiSnapshot { step: 0, filtration: 0.0, betti: vec![3, 0] },
                BettiSnapshot { step: 1, filtration: 1.0, betti: vec![2, 1] },
                BettiSnapshot { step: 2, filtration: 2.0, betti: vec![1, 0] },
            ],
            1,
        );
        assert_eq!(curve.max_betti0(), 3);
    }

    #[test]
    fn test_betti_curve_max_betti1() {
        let curve = BettiCurve::new(
            vec![
                BettiSnapshot { step: 0, filtration: 0.0, betti: vec![3, 0] },
                BettiSnapshot { step: 1, filtration: 1.0, betti: vec![2, 2] },
                BettiSnapshot { step: 2, filtration: 2.0, betti: vec![1, 1] },
            ],
            1,
        );
        assert_eq!(curve.max_betti1(), 2);
    }

    #[test]
    fn test_betti_curve_area() {
        let curve = BettiCurve::new(
            vec![
                BettiSnapshot { step: 0, filtration: 0.0, betti: vec![2, 0] },
                BettiSnapshot { step: 10, filtration: 1.0, betti: vec![2, 0] },
            ],
            1,
        );
        assert!((curve.betti0_area() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_betti1_area() {
        let curve = BettiCurve::new(
            vec![
                BettiSnapshot { step: 0, filtration: 0.0, betti: vec![1, 3] },
                BettiSnapshot { step: 5, filtration: 1.0, betti: vec![1, 1] },
            ],
            1,
        );
        // Trapezoid: (3 + 1) / 2 * 5 = 10
        assert!((curve.betti1_area() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_betti_at_step() {
        let curve = BettiCurve::new(
            vec![
                BettiSnapshot { step: 0, filtration: 0.0, betti: vec![2, 1] },
                BettiSnapshot { step: 5, filtration: 1.0, betti: vec![1, 0] },
            ],
            1,
        );
        assert_eq!(curve.betti_at_step(0, 0), Some(2));
        assert_eq!(curve.betti_at_step(5, 1), Some(0));
        assert_eq!(curve.betti_at_step(99, 0), None);
    }

    #[test]
    fn test_euler_characteristic_curve() {
        let curve = BettiCurve::new(
            vec![
                BettiSnapshot { step: 0, filtration: 0.0, betti: vec![3, 1] },
                BettiSnapshot { step: 1, filtration: 1.0, betti: vec![1, 0] },
            ],
            1,
        );
        let euler = euler_characteristic_curve(&curve);
        assert_eq!(euler[0].1, 2); // 3 - 1 = 2
        assert_eq!(euler[1].1, 1); // 1 - 0 = 1
    }

    #[test]
    fn test_betti_snapshot_serde() {
        let snap = BettiSnapshot { step: 5, filtration: 1.5, betti: vec![2, 1] };
        let json = serde_json::to_string(&snap).unwrap();
        let snap2: BettiSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snap2.step, 5);
        assert_eq!(snap2.betti, vec![2, 1]);
    }

    #[test]
    fn test_betti_curve_serde() {
        let curve = BettiCurve::new(
            vec![BettiSnapshot { step: 0, filtration: 0.0, betti: vec![1] }],
            0,
        );
        let json = serde_json::to_string(&curve).unwrap();
        let curve2: BettiCurve = serde_json::from_str(&json).unwrap();
        assert_eq!(curve2.snapshots.len(), 1);
    }
}
