//! Persistence diagram computation via Vietoris-Rips complex.

use crate::belief::StateHistory;
use serde::{Deserialize, Serialize};

/// A single point in a persistence diagram (birth, death, dimension).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistencePair {
    pub birth: f64,
    pub death: f64,
    pub dim: usize,
}

impl PersistencePair {
    pub fn new(birth: f64, death: f64, dim: usize) -> Self {
        Self { birth, death, dim }
    }

    /// How long the feature persists.
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Midpoint of the feature's lifespan.
    pub fn midpoint(&self) -> f64 {
        (self.birth + self.death) / 2.0
    }

    /// Is this a finite feature (not infinite)?
    pub fn is_finite(&self) -> bool {
        self.death.is_finite()
    }
}

/// A persistence diagram: collection of birth-death pairs across dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
    pub n_points: usize,
}

impl PersistenceDiagram {
    pub fn new(pairs: Vec<PersistencePair>, n_points: usize) -> Self {
        Self { pairs, n_points }
    }

    /// Empty diagram.
    pub fn empty() -> Self {
        Self {
            pairs: vec![],
            n_points: 0,
        }
    }

    /// Filter pairs by homology dimension.
    pub fn dimension(&self, dim: usize) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| p.dim == dim).collect()
    }

    /// Maximum persistence value.
    pub fn max_persistence(&self) -> f64 {
        self.pairs
            .iter()
            .filter(|p| p.is_finite())
            .map(|p| p.persistence())
            .fold(0.0_f64, f64::max)
    }

    /// Total persistence (sum of all persistences, optionally raised to power).
    pub fn total_persistence(&self, power: f64) -> f64 {
        self.pairs
            .iter()
            .filter(|p| p.is_finite())
            .map(|p| p.persistence().powf(power))
            .sum()
    }

    /// Number of features with persistence above a threshold.
    pub fn significant_features(&self, threshold: f64) -> usize {
        self.pairs.iter().filter(|p| p.persistence() > threshold).count()
    }

    /// Average persistence of all features.
    pub fn avg_persistence(&self) -> f64 {
        let finite: Vec<_> = self.pairs.iter().filter(|p| p.is_finite()).collect();
        if finite.is_empty() {
            return 0.0;
        }
        finite.iter().map(|p| p.persistence()).sum::<f64>() / finite.len() as f64
    }

    /// Persistence entropy — measures the uniformity of the persistence distribution.
    pub fn persistence_entropy(&self) -> f64 {
        if self.pairs.is_empty() {
            return 0.0;
        }
        let total = self.total_persistence(1.0);
        if total <= 0.0 {
            return 0.0;
        }
        let entropy: f64 = self
            .pairs
            .iter()
            .map(|p| {
                let p_i = p.persistence() / total;
                if p_i > 0.0 { -p_i * p_i.ln() } else { 0.0 }
            })
            .sum();
        entropy
    }
}

/// Compute a Vietoris-Rips persistence diagram from a state history.
///
/// This uses a simple distance-based approach. For a full implementation,
/// one would use a library like `ripser`, but here we implement a simplified
/// version that computes H0 (connected components) and H1 (loops) persistence.
pub fn compute_persistence(history: &StateHistory, max_dim: usize) -> PersistenceDiagram {
    if history.len() < 2 {
        return PersistenceDiagram::empty();
    }

    let dist = history.distance_matrix();
    let n = history.len();
    let mut pairs = Vec::new();

    // H0: connected components via single-linkage clustering
    let h0_pairs = compute_h0(&dist, n);
    pairs.extend(h0_pairs);

    // H1: loops (simplified via distance matrix analysis)
    if max_dim >= 1 {
        let h1_pairs = compute_h1(&dist, n);
        pairs.extend(h1_pairs);
    }

    PersistenceDiagram::new(pairs, n)
}

/// Compute H0 persistence (connected components) using union-find + sorted edges.
fn compute_h0(dist: &nalgebra::DMatrix<f64>, n: usize) -> Vec<PersistencePair> {
    // Collect all edges sorted by distance
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((dist[(i, j)], i, j));
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank = vec![0usize; n];
    let mut components = n;
    let mut pairs = Vec::new();

    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    for (d, u, v) in edges {
        let ru = find(&mut parent, u);
        let rv = find(&mut parent, v);
        if ru != rv {
            // Birth at 0.0 (all points start as separate components), death when merged
            pairs.push(PersistencePair::new(0.0, d, 0));
            if rank[ru] < rank[rv] {
                parent[ru] = rv;
            } else if rank[ru] > rank[rv] {
                parent[rv] = ru;
            } else {
                parent[rv] = ru;
                rank[ru] += 1;
            }
            components -= 1;
            if components == 1 {
                break;
            }
        }
    }

    // Add the essential component (born at 0, never dies)
    pairs.push(PersistencePair::new(0.0, f64::INFINITY, 0));

    pairs
}

/// Compute H1 persistence (1-dimensional loops) via simplified approach.
///
/// For a proper Vietoris-Rips H1 computation, we look for "shortcuts" in
/// triangles — when a triangle is formed, we check if it creates a cycle.
fn compute_h1(dist: &nalgebra::DMatrix<f64>, n: usize) -> Vec<PersistencePair> {
    let mut pairs = Vec::new();

    if n < 3 {
        return pairs;
    }

    // For each triangle (i, j, k), the H1 feature is born at the edge
    // that completes the triangle and dies when the triangle fills.
    // Birth = max(d_ij, d_jk, d_ik), Death = max(d_ij, d_jk, d_ik)
    // In the Rips complex, a triangle forms at filtration value max(d_ij, d_jk, d_ik).
    // A cycle appears at the third shortest edge, and dies when the triangle fills.

    // We use a simplified model: for each triple, birth = third shortest edge,
    // death = longest edge. But only if this creates a new cycle.
    let mut all_triples: Vec<(f64, f64)> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let d_ij = dist[(i, j)];
                let d_jk = dist[(j, k)];
                let d_ik = dist[(i, k)];
                let mut edges = [d_ij, d_jk, d_ik];
                edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                // H1: born when third edge enters (edges[1]), dies when triangle fills (edges[2])
                if edges[2] > edges[1] {
                    all_triples.push((edges[1], edges[2]));
                }
            }
        }
    }

    // Sort by birth time
    all_triples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take the most persistent loops (simplified: we don't track exact boundary maps)
    // Limit to avoid overcounting
    let max_loops = (n / 3).max(1);
    for &(birth, death) in all_triples.iter().take(max_loops) {
        pairs.push(PersistencePair::new(birth, death, 1));
    }

    pairs
}

/// Compute persistence diagram for a sliding window over the state history.
pub fn sliding_window_persistence(
    history: &StateHistory,
    window_size: usize,
    stride: usize,
    max_dim: usize,
) -> Vec<(usize, PersistenceDiagram)> {
    let mut results = Vec::new();
    let mut start = 0;
    while start + window_size <= history.len() {
        let window = StateHistory {
            states: history.states[start..start + window_size].to_vec(),
        };
        let dg = compute_persistence(&window, max_dim);
        results.push((start, dg));
        start += stride;
    }
    results
}

/// Bottleneck distance between two persistence diagrams.
pub fn bottleneck_distance(dg1: &PersistenceDiagram, dg2: &PersistenceDiagram) -> f64 {
    // Simplified: for each pair in dg1, find closest pair in dg2 (same dimension)
    let mut max_dist = 0.0_f64;
    for p1 in &dg1.pairs {
        if !p1.is_finite() {
            continue;
        }
        let best = dg2
            .pairs
            .iter()
            .filter(|p| p.dim == p1.dim && p.is_finite())
            .map(|p2| {
                ((p1.birth - p2.birth).powi(2) + (p1.death - p2.death).powi(2)).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        // Distance to diagonal
        let diag_dist = (p1.birth - p1.death).abs() / std::f64::consts::SQRT_2;
        max_dist = max_dist.max(best.min(diag_dist));
    }
    max_dist
}

/// Wasserstein distance between two persistence diagrams (p=2).
pub fn wasserstein_distance(dg1: &PersistenceDiagram, dg2: &PersistenceDiagram, p: f64) -> f64 {
    let mut total = 0.0;
    for p1 in &dg1.pairs {
        if !p1.is_finite() {
            continue;
        }
        let best = dg2
            .pairs
            .iter()
            .filter(|p| p.dim == p1.dim && p.is_finite())
            .map(|p2| {
                ((p1.birth - p2.birth).powi(2) + (p1.death - p2.death).powi(2)).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        let diag_dist = (p1.birth - p1.death).abs() / std::f64::consts::SQRT_2;
        total += best.min(diag_dist).powf(p);
    }
    total.powf(1.0 / p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::*;

    #[test]
    fn test_persistence_pair() {
        let pp = PersistencePair::new(0.5, 2.0, 1);
        assert!((pp.persistence() - 1.5).abs() < 1e-10);
        assert!((pp.midpoint() - 1.25).abs() < 1e-10);
        assert!(pp.is_finite());
    }

    #[test]
    fn test_persistence_pair_infinite() {
        let pp = PersistencePair::new(0.0, f64::INFINITY, 0);
        assert!(!pp.is_finite());
        assert!(pp.persistence().is_infinite());
    }

    #[test]
    fn test_empty_diagram() {
        let dg = PersistenceDiagram::empty();
        assert_eq!(dg.pairs.len(), 0);
        assert_eq!(dg.max_persistence(), 0.0);
        assert_eq!(dg.avg_persistence(), 0.0);
    }

    #[test]
    fn test_diagram_max_persistence() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0),
                PersistencePair::new(0.0, 3.0, 0),
                PersistencePair::new(0.5, 2.0, 1),
            ],
            5,
        );
        assert!((dg.max_persistence() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_diagram_total_persistence() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0),
                PersistencePair::new(0.0, 3.0, 0),
            ],
            3,
        );
        assert!((dg.total_persistence(1.0) - 4.0).abs() < 1e-10);
        assert!((dg.total_persistence(2.0) - (1.0 + 9.0)).abs() < 1e-10);
    }

    #[test]
    fn test_diagram_significant_features() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 0.5, 0),
                PersistencePair::new(0.0, 3.0, 0),
                PersistencePair::new(0.5, 2.0, 1),
            ],
            4,
        );
        assert_eq!(dg.significant_features(1.0), 2);
        assert_eq!(dg.significant_features(5.0), 0);
    }

    #[test]
    fn test_diagram_avg_persistence() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 2.0, 0),
                PersistencePair::new(0.0, 4.0, 0),
            ],
            3,
        );
        assert!((dg.avg_persistence() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_diagram_dimension_filter() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0),
                PersistencePair::new(0.0, 2.0, 1),
                PersistencePair::new(0.5, 3.0, 1),
            ],
            4,
        );
        assert_eq!(dg.dimension(0).len(), 1);
        assert_eq!(dg.dimension(1).len(), 2);
        assert_eq!(dg.dimension(2).len(), 0);
    }

    #[test]
    fn test_persistence_entropy() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0),
                PersistencePair::new(0.0, 1.0, 0),
            ],
            3,
        );
        // Equal persistences → maximum entropy for 2 items
        let e = dg.persistence_entropy();
        assert!(e > 0.0);
        assert!((e - (2.0_f64).ln()).abs() < 1e-6);
    }

    #[test]
    fn test_persistence_entropy_empty() {
        let dg = PersistenceDiagram::empty();
        assert_eq!(dg.persistence_entropy(), 0.0);
    }

    #[test]
    fn test_compute_persistence_basic() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0], 0),
            BeliefState::new(vec![1.0], 1),
            BeliefState::new(vec![2.0], 2),
        ]);
        let dg = compute_persistence(&h, 1);
        assert!(dg.pairs.len() > 0);
        // Should have H0 pairs
        assert!(dg.dimension(0).len() > 0);
    }

    #[test]
    fn test_compute_persistence_single_point() {
        let h = StateHistory::new(vec![BeliefState::new(vec![0.0], 0)]);
        let dg = compute_persistence(&h, 1);
        assert_eq!(dg.pairs.len(), 0);
    }

    #[test]
    fn test_compute_persistence_two_points() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0], 0),
            BeliefState::new(vec![1.0], 1),
        ]);
        let dg = compute_persistence(&h, 1);
        assert!(dg.pairs.len() > 0);
    }

    #[test]
    fn test_h0_clustered_points() {
        // Two clusters: {0,1} and {10,11}
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0], 0),
            BeliefState::new(vec![1.0], 1),
            BeliefState::new(vec![10.0], 2),
            BeliefState::new(vec![11.0], 3),
        ]);
        let dg = compute_persistence(&h, 0);
        let h0: Vec<_> = dg.dimension(0);
        // Should have one pair with large persistence (inter-cluster distance)
        let max_p = h0.iter().filter(|p| p.is_finite()).map(|p| p.persistence()).fold(0.0_f64, f64::max);
        assert!(max_p > 5.0, "Expected large inter-cluster persistence, got {}", max_p);
    }

    #[test]
    fn test_sliding_window() {
        let t = synthetic_trajectory(2, 50, 0.1, 0.01, 42);
        let windows = sliding_window_persistence(&t, 10, 5, 1);
        assert!(windows.len() >= 8);
        for (start, dg) in &windows {
            assert!(*start < 50);
            assert!(dg.n_points == 10);
        }
    }

    #[test]
    fn test_bottleneck_distance_same() {
        let dg = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 1.0, 0)],
            3,
        );
        let d = bottleneck_distance(&dg, &dg);
        assert!(d < 1e-10);
    }

    #[test]
    fn test_bottleneck_distance_different() {
        let dg1 = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 1.0, 0)],
            3,
        );
        let dg2 = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 5.0, 0)],
            3,
        );
        let d = bottleneck_distance(&dg1, &dg2);
        assert!(d > 0.0);
    }

    #[test]
    fn test_wasserstein_distance_same() {
        let dg = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 1.0, 0)],
            3,
        );
        let d = wasserstein_distance(&dg, &dg, 2.0);
        assert!(d < 1e-10);
    }

    #[test]
    fn test_wasserstein_distance_different() {
        let dg1 = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 1.0, 0)],
            3,
        );
        let dg2 = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 5.0, 0)],
            3,
        );
        let d = wasserstein_distance(&dg1, &dg2, 2.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_diagram_serde_roundtrip() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0),
                PersistencePair::new(0.5, 2.0, 1),
            ],
            5,
        );
        let json = serde_json::to_string(&dg).unwrap();
        let dg2: PersistenceDiagram = serde_json::from_str(&json).unwrap();
        assert_eq!(dg2.pairs.len(), 2);
        assert!((dg2.pairs[0].birth - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_persistence_identical_points() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![1.0], 0),
            BeliefState::new(vec![1.0], 1),
            BeliefState::new(vec![1.0], 2),
        ]);
        let dg = compute_persistence(&h, 1);
        // All distances are 0, so all H0 pairs merge at 0
        let finite_h0: Vec<_> = dg.dimension(0).into_iter().filter(|p| p.is_finite()).collect();
        for p in &finite_h0 {
            assert!(p.persistence() < 1e-10);
        }
    }

    #[test]
    fn test_persistence_pair_midpoint() {
        let pp = PersistencePair::new(1.0, 5.0, 0);
        assert!((pp.midpoint() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_h1_with_loop() {
        // Create points roughly on a circle
        let n = 8;
        let states: Vec<BeliefState> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                BeliefState::new(vec![angle.cos(), angle.sin()], i)
            })
            .collect();
        let h = StateHistory::new(states);
        let dg = compute_persistence(&h, 1);
        let h1 = dg.dimension(1);
        // A circular arrangement should produce at least one H1 feature
        assert!(h1.len() > 0, "Expected at least one H1 feature for circular points");
    }
}
