//! Belief manifold construction from agent state history.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A single agent belief state — a point in the belief manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    /// Coordinates in belief space.
    pub coords: Vec<f64>,
    /// Optional timestamp or learning step.
    pub step: usize,
}

impl BeliefState {
    pub fn new(coords: Vec<f64>, step: usize) -> Self {
        Self { coords, step }
    }

    pub fn dimension(&self) -> usize {
        self.coords.len()
    }

    /// Euclidean distance to another belief state.
    pub fn distance_to(&self, other: &BeliefState) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Convert to nalgebra column vector.
    pub fn to_vector(&self) -> nalgebra::DVector<f64> {
        nalgebra::DVector::from_vec(self.coords.clone())
    }
}

/// History of agent states over learning steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateHistory {
    pub states: Vec<BeliefState>,
}

impl StateHistory {
    pub fn new(states: Vec<BeliefState>) -> Self {
        Self { states }
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn dimension(&self) -> Option<usize> {
        self.states.first().map(|s| s.dimension())
    }

    /// Get states within a step range [lo, hi).
    pub fn slice_by_step(&self, lo: usize, hi: usize) -> StateHistory {
        StateHistory {
            states: self
                .states
                .iter()
                .filter(|s| s.step >= lo && s.step < hi)
                .cloned()
                .collect(),
        }
    }

    /// Compute pairwise distance matrix.
    pub fn distance_matrix(&self) -> DMatrix<f64> {
        let n = self.states.len();
        let mut mat = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let d = self.states[i].distance_to(&self.states[j]);
                mat[(i, j)] = d;
                mat[(j, i)] = d;
            }
        }
        mat
    }

    /// Compute centroid of all belief states.
    pub fn centroid(&self) -> Option<BeliefState> {
        if self.states.is_empty() {
            return None;
        }
        let dim = self.states[0].dimension();
        let mut sum = vec![0.0; dim];
        for s in &self.states {
            for (i, c) in s.coords.iter().enumerate() {
                sum[i] += c;
            }
        }
        let n = self.states.len() as f64;
        let avg: Vec<f64> = sum.iter().map(|v| v / n).collect();
        Some(BeliefState::new(avg, 0))
    }

    /// Variance of belief states around centroid.
    pub fn spread(&self) -> f64 {
        let centroid = match self.centroid() {
            Some(c) => c,
            None => return 0.0,
        };
        let n = self.states.len() as f64;
        self.states
            .iter()
            .map(|s| s.distance_to(&centroid).powi(2))
            .sum::<f64>()
            / n
    }
}

/// Generate a synthetic belief trajectory.
pub fn synthetic_trajectory(
    dim: usize,
    steps: usize,
    drift: f64,
    noise: f64,
    seed: u64,
) -> StateHistory {
    let mut rng = simple_rng(seed);
    let mut pos = vec![0.0; dim];
    let mut states = Vec::with_capacity(steps);

    for step in 0..steps {
        for c in pos.iter_mut() {
            *c += drift * (random_f64(&mut rng) - 0.5) * 2.0;
            *c += noise * (random_f64(&mut rng) - 0.5) * 2.0;
        }
        states.push(BeliefState::new(pos.clone(), step));
    }
    StateHistory::new(states)
}

/// Generate a spiral belief trajectory (converging to a point).
pub fn spiral_trajectory(dim: usize, steps: usize, radius: f64, noise: f64, seed: u64) -> StateHistory {
    let mut rng = simple_rng(seed);
    let mut states = Vec::with_capacity(steps);
    for step in 0..steps {
        let t = step as f64 / steps as f64;
        let angle = t * 4.0 * std::f64::consts::PI;
        let r = radius * (1.0 - t);
        let mut coords = vec![0.0; dim];
        if dim >= 2 {
            coords[0] = r * angle.cos() + noise * (random_f64(&mut rng) - 0.5);
            coords[1] = r * angle.sin() + noise * (random_f64(&mut rng) - 0.5);
        }
        for i in 2..dim {
            coords[i] = noise * (random_f64(&mut rng) - 0.5) * 0.1;
        }
        states.push(BeliefState::new(coords, step));
    }
    StateHistory::new(states)
}

/// Generate a clustered belief trajectory (jumps between clusters).
pub fn clustered_trajectory(
    dim: usize,
    steps: usize,
    n_clusters: usize,
    cluster_spread: f64,
    noise: f64,
    seed: u64,
) -> StateHistory {
    let mut rng = simple_rng(seed);
    let mut centers = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        let mut c = Vec::with_capacity(dim);
        for _ in 0..dim {
            c.push((random_f64(&mut rng) - 0.5) * 2.0 * cluster_spread);
        }
        centers.push(c);
    }

    let steps_per_cluster = steps / n_clusters;
    let mut states = Vec::with_capacity(steps);
    for (ci, center) in centers.iter().enumerate() {
        let start = ci * steps_per_cluster;
        let end = if ci == n_clusters - 1 { steps } else { start + steps_per_cluster };
        for step in start..end {
            let mut coords = Vec::with_capacity(dim);
            for (i, c_val) in center.iter().enumerate() {
                coords.push(*c_val + noise * (random_f64(&mut rng) - 0.5) * 2.0);
            }
            states.push(BeliefState::new(coords, step));
        }
    }
    StateHistory::new(states)
}

// Simple LCG PRNG for reproducibility
fn simple_rng(seed: u64) -> u64 {
    seed
}

fn random_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_state_creation() {
        let bs = BeliefState::new(vec![1.0, 2.0, 3.0], 5);
        assert_eq!(bs.dimension(), 3);
        assert_eq!(bs.step, 5);
    }

    #[test]
    fn test_belief_state_distance_self() {
        let bs = BeliefState::new(vec![1.0, 2.0], 0);
        assert!((bs.distance_to(&bs) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_belief_state_distance_known() {
        let a = BeliefState::new(vec![0.0, 0.0], 0);
        let b = BeliefState::new(vec![3.0, 4.0], 1);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_belief_state_distance_symmetry() {
        let a = BeliefState::new(vec![1.0, 2.0, 3.0], 0);
        let b = BeliefState::new(vec![4.0, 5.0, 6.0], 1);
        assert!((a.distance_to(&b) - b.distance_to(&a)).abs() < 1e-10);
    }

    #[test]
    fn test_state_history_len() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0], 0),
            BeliefState::new(vec![1.0], 1),
        ]);
        assert_eq!(h.len(), 2);
        assert!(!h.is_empty());
    }

    #[test]
    fn test_state_history_empty() {
        let h = StateHistory::new(vec![]);
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn test_state_history_dimension() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![1.0, 2.0], 0),
        ]);
        assert_eq!(h.dimension(), Some(2));
    }

    #[test]
    fn test_state_history_dimension_empty() {
        let h = StateHistory::new(vec![]);
        assert_eq!(h.dimension(), None);
    }

    #[test]
    fn test_distance_matrix_diagonal_zero() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![1.0], 0),
            BeliefState::new(vec![2.0], 1),
        ]);
        let dm = h.distance_matrix();
        assert!((dm[(0, 0)] - 0.0).abs() < 1e-10);
        assert!((dm[(1, 1)] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_matrix_symmetric() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0, 0.0], 0),
            BeliefState::new(vec![1.0, 0.0], 1),
            BeliefState::new(vec![0.0, 1.0], 2),
        ]);
        let dm = h.distance_matrix();
        assert!((dm[(0, 1)] - dm[(1, 0)]).abs() < 1e-10);
        assert!((dm[(0, 2)] - dm[(2, 0)]).abs() < 1e-10);
        assert!((dm[(1, 2)] - dm[(2, 1)]).abs() < 1e-10);
    }

    #[test]
    fn test_centroid() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0, 0.0], 0),
            BeliefState::new(vec![2.0, 4.0], 1),
        ]);
        let c = h.centroid().unwrap();
        assert!((c.coords[0] - 1.0).abs() < 1e-10);
        assert!((c.coords[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_empty() {
        let h = StateHistory::new(vec![]);
        assert!(h.centroid().is_none());
    }

    #[test]
    fn test_spread_zero() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![1.0], 0),
            BeliefState::new(vec![1.0], 1),
        ]);
        assert!(h.spread() < 1e-10);
    }

    #[test]
    fn test_spread_positive() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0], 0),
            BeliefState::new(vec![2.0], 1),
        ]);
        assert!(h.spread() > 0.0);
    }

    #[test]
    fn test_slice_by_step() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0], 0),
            BeliefState::new(vec![1.0], 1),
            BeliefState::new(vec![2.0], 2),
            BeliefState::new(vec![3.0], 3),
        ]);
        let s = h.slice_by_step(1, 3);
        assert_eq!(s.len(), 2);
        assert_eq!(s.states[0].step, 1);
        assert_eq!(s.states[1].step, 2);
    }

    #[test]
    fn test_synthetic_trajectory_length() {
        let t = synthetic_trajectory(3, 50, 0.1, 0.01, 42);
        assert_eq!(t.len(), 50);
    }

    #[test]
    fn test_synthetic_trajectory_dimension() {
        let t = synthetic_trajectory(5, 10, 0.1, 0.01, 42);
        assert_eq!(t.dimension(), Some(5));
    }

    #[test]
    fn test_synthetic_trajectory_reproducible() {
        let t1 = synthetic_trajectory(3, 10, 0.1, 0.01, 42);
        let t2 = synthetic_trajectory(3, 10, 0.1, 0.01, 42);
        for (a, b) in t1.states.iter().zip(t2.states.iter()) {
            assert!((a.coords[0] - b.coords[0]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spiral_trajectory() {
        let t = spiral_trajectory(2, 100, 5.0, 0.01, 42);
        assert_eq!(t.len(), 100);
        // First point should be roughly at radius
        let first = &t.states[0];
        let r0 = (first.coords[0].powi(2) + first.coords[1].powi(2)).sqrt();
        assert!(r0 > 1.0); // not at origin
    }

    #[test]
    fn test_spiral_converges() {
        let t = spiral_trajectory(2, 200, 5.0, 0.0, 42);
        let first_r = {
            let s = &t.states[0];
            (s.coords[0].powi(2) + s.coords[1].powi(2)).sqrt()
        };
        let last_r = {
            let s = t.states.last().unwrap();
            (s.coords[0].powi(2) + s.coords[1].powi(2)).sqrt()
        };
        assert!(last_r < first_r);
    }

    #[test]
    fn test_clustered_trajectory() {
        let t = clustered_trajectory(3, 60, 3, 5.0, 0.5, 42);
        assert_eq!(t.len(), 60);
    }

    #[test]
    fn test_to_vector() {
        let bs = BeliefState::new(vec![1.0, 2.0, 3.0], 0);
        let v = bs.to_vector();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_history_serde_roundtrip() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![1.0, 2.0], 0),
            BeliefState::new(vec![3.0, 4.0], 1),
        ]);
        let json = serde_json::to_string(&h).unwrap();
        let h2: StateHistory = serde_json::from_str(&json).unwrap();
        assert_eq!(h2.len(), 2);
        assert!((h2.states[0].coords[0] - 1.0).abs() < 1e-10);
    }
}
