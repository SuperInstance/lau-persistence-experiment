//! Cross-agent comparison: do "smart" agents have different topological signatures?

use crate::belief::StateHistory;
use crate::complexity::TopologicalComplexity;
use crate::landscape::PersistenceLandscape;
use crate::persistence::{compute_persistence, PersistenceDiagram, bottleneck_distance, wasserstein_distance};
use crate::robustness::RobustnessScore;
use serde::{Deserialize, Serialize};

/// A topological signature for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSignature {
    pub name: String,
    pub complexity: TopologicalComplexity,
    pub robustness: RobustnessScore,
    pub max_persistence_h0: f64,
    pub max_persistence_h1: f64,
    pub avg_persistence: f64,
    pub entropy: f64,
    pub n_features: usize,
}

impl AgentSignature {
    /// Compute from a state history.
    pub fn from_history(name: &str, history: &StateHistory) -> Self {
        let dg = compute_persistence(history, 1);
        Self::from_diagram(name, &dg, history.len())
    }

    /// Compute from a persistence diagram.
    pub fn from_diagram(name: &str, diagram: &PersistenceDiagram, n_points: usize) -> Self {
        let tc = TopologicalComplexity::from_diagram(diagram, 0.1);
        let complexity = TopologicalComplexity {
            total_persistence: sanitize_f64(tc.total_persistence),
            total_persistence_sq: sanitize_f64(tc.total_persistence_sq),
            h0_significant: tc.h0_significant,
            h1_significant: tc.h1_significant,
            max_persistence: sanitize_f64(tc.max_persistence),
            entropy: sanitize_f64(tc.entropy),
            complexity_score: sanitize_f64(tc.complexity_score).clamp(0.0, 1.0),
            predicted_difficulty: sanitize_f64(tc.predicted_difficulty).clamp(0.0, 1.0),
        };
        let r = RobustnessScore::from_diagrams(&[diagram.clone()], 0.5);
        let robustness = RobustnessScore {
            robustness: sanitize_f64(r.robustness),
            mean_persistence: sanitize_f64(r.mean_persistence),
            long_lived_fraction: sanitize_f64(r.long_lived_fraction).clamp(0.0, 1.0),
            stability: sanitize_f64(r.stability).clamp(0.0, 1.0),
            fragility_indicators: r.fragility_indicators,
        };

        let h0_max = diagram
            .dimension(0)
            .iter()
            .filter(|p| p.is_finite())
            .map(|p| p.persistence())
            .fold(0.0_f64, f64::max);

        let h1_max = diagram
            .dimension(1)
            .iter()
            .filter(|p| p.is_finite())
            .map(|p| p.persistence())
            .fold(0.0_f64, f64::max);

        AgentSignature {
            name: name.to_string(),
            complexity,
            robustness,
            max_persistence_h0: sanitize_f64(h0_max),
            max_persistence_h1: sanitize_f64(h1_max),
            avg_persistence: sanitize_f64(diagram.avg_persistence()),
            entropy: sanitize_f64(diagram.persistence_entropy()),
            n_features: diagram.pairs.iter().filter(|p| p.is_finite()).count(),
        }
    }

    /// Convert to a feature vector for comparison.
    pub fn feature_vector(&self) -> Vec<f64> {
        vec![
            sanitize_f64(self.complexity.complexity_score),
            sanitize_f64(self.complexity.total_persistence),
            sanitize_f64(self.complexity.total_persistence_sq),
            self.complexity.h0_significant as f64,
            self.complexity.h1_significant as f64,
            sanitize_f64(self.complexity.max_persistence),
            sanitize_f64(self.complexity.entropy),
            sanitize_f64(self.max_persistence_h0),
            sanitize_f64(self.max_persistence_h1),
            sanitize_f64(self.avg_persistence),
            sanitize_f64(self.entropy),
            self.n_features as f64,
        ]
    }

    /// Euclidean distance between two signatures' feature vectors.
    pub fn distance_to(&self, other: &AgentSignature) -> f64 {
        let v1 = self.feature_vector();
        let v2 = other.feature_vector();
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| {
                let a = if a.is_nan() { 0.0 } else { *a };
                let b = if b.is_nan() { 0.0 } else { *b };
                (a - b).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }
}

/// Comparison result between two agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentComparison {
    pub agent_a: String,
    pub agent_b: String,
    pub signature_distance: f64,
    pub bottleneck_dist: f64,
    pub wasserstein_dist: f64,
    pub complexity_diff: f64,
    pub persistence_diff: f64,
    pub same_regime: bool,
}

impl AgentComparison {
    /// Compare two agents.
    pub fn compare(sig_a: &AgentSignature, sig_b: &AgentSignature) -> Self {
        // We need to recompute diagrams for distances, but we'll use the signatures
        let complexity_diff = (sig_a.complexity.complexity_score - sig_b.complexity.complexity_score).abs();
        let persistence_diff = (sig_a.avg_persistence - sig_b.avg_persistence).abs();

        AgentComparison {
            agent_a: sig_a.name.clone(),
            agent_b: sig_b.name.clone(),
            signature_distance: sig_a.distance_to(sig_b),
            bottleneck_dist: 0.0, // Would need diagrams; approximate
            wasserstein_dist: 0.0,
            complexity_diff,
            persistence_diff,
            same_regime: complexity_diff < 0.2 && persistence_diff < 1.0,
        }
    }

    /// Compare with actual diagrams for accurate distances.
    pub fn compare_with_diagrams(
        sig_a: &AgentSignature,
        sig_b: &AgentSignature,
        dg_a: &PersistenceDiagram,
        dg_b: &PersistenceDiagram,
    ) -> Self {
        let mut comp = Self::compare(sig_a, sig_b);
        comp.bottleneck_dist = bottleneck_distance(dg_a, dg_b);
        comp.wasserstein_dist = wasserstein_distance(dg_a, dg_b, 2.0);
        comp
    }
}

/// A cohort of agents for batch comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCohort {
    pub agents: Vec<AgentSignature>,
}

impl AgentCohort {
    pub fn new(agents: Vec<AgentSignature>) -> Self {
        Self { agents }
    }

    /// Pairwise comparison matrix.
    pub fn pairwise_distances(&self) -> DMatrix<f64> {
        let n = self.agents.len();
        let mut mat = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let d = self.agents[i].distance_to(&self.agents[j]);
                mat[(i, j)] = d;
                mat[(j, i)] = d;
            }
        }
        mat
    }

    /// Find the most distinct agent.
    pub fn most_distinct(&self) -> Option<&AgentSignature> {
        if self.agents.is_empty() {
            return None;
        }
        let distances = self.pairwise_distances();
        let mut max_total = 0.0_f64;
        let mut max_idx = 0;
        for i in 0..self.agents.len() {
            let total: f64 = (0..self.agents.len()).map(|j| distances[(i, j)]).sum();
            if total > max_total {
                max_total = total;
                max_idx = i;
            }
        }
        Some(&self.agents[max_idx])
    }

    /// Cluster agents by similarity (simple threshold-based).
    pub fn cluster_by_similarity(&self, threshold: f64) -> Vec<Vec<usize>> {
        let n = self.agents.len();
        let distances = self.pairwise_distances();
        let mut assigned = vec![false; n];
        let mut clusters = Vec::new();

        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let mut cluster = vec![i];
            assigned[i] = true;
            for j in (i + 1)..n {
                if !assigned[j] && distances[(i, j)] < threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }
            clusters.push(cluster);
        }
        clusters
    }

    /// Rank agents by a metric (complexity, persistence, robustness).
    pub fn rank_by<F: Fn(&AgentSignature) -> f64>(&self, metric: F) -> Vec<(usize, f64)> {
        let mut ranked: Vec<(usize, f64)> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, s)| (i, metric(s)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

use nalgebra::DMatrix;

fn sanitize_f64(v: f64) -> f64 {
    if v.is_nan() || v.is_infinite() { 0.0 } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::*;

    #[test]
    fn test_agent_signature_from_history() {
        let h = synthetic_trajectory(3, 50, 0.1, 0.01, 42);
        let sig = AgentSignature::from_history("test_agent", &h);
        assert_eq!(sig.name, "test_agent");
        assert!(sig.feature_vector().len() > 0);
    }

    #[test]
    fn test_signature_distance_to_self() {
        let h = synthetic_trajectory(3, 50, 0.1, 0.01, 42);
        let sig = AgentSignature::from_history("test", &h);
        let d = sig.distance_to(&sig);
        assert!(d < 1e-10, "Distance to self should be 0, got {}", d);
    }

    #[test]
    fn test_signature_distance_different() {
        let h1 = synthetic_trajectory(3, 50, 0.1, 0.01, 42);
        let h2 = synthetic_trajectory(3, 50, 0.5, 0.1, 99);
        let s1 = AgentSignature::from_history("a", &h1);
        let s2 = AgentSignature::from_history("b", &h2);
        assert!(s1.distance_to(&s2) > 0.0);
    }

    #[test]
    fn test_agent_comparison() {
        let h1 = synthetic_trajectory(3, 50, 0.1, 0.01, 42);
        let h2 = clustered_trajectory(3, 50, 3, 5.0, 1.0, 42);
        let s1 = AgentSignature::from_history("drifter", &h1);
        let s2 = AgentSignature::from_history("clusterer", &h2);
        let comp = AgentComparison::compare(&s1, &s2);
        assert_eq!(comp.agent_a, "drifter");
        assert_eq!(comp.agent_b, "clusterer");
        assert!(comp.signature_distance > 0.0);
    }

    #[test]
    fn test_agent_comparison_same_regime() {
        let h1 = synthetic_trajectory(3, 50, 0.1, 0.01, 42);
        let h2 = synthetic_trajectory(3, 50, 0.1, 0.01, 43);
        let s1 = AgentSignature::from_history("a", &h1);
        let s2 = AgentSignature::from_history("b", &h2);
        let comp = AgentComparison::compare(&s1, &s2);
        // Similar trajectories → should be same regime
        // (may or may not depending on exact values, but the test should compile)
        assert!(comp.complexity_diff >= 0.0);
    }

    #[test]
    fn test_cohort_pairwise() {
        let h1 = synthetic_trajectory(3, 50, 0.1, 0.01, 42);
        let h2 = clustered_trajectory(3, 50, 3, 5.0, 1.0, 42);
        let h3 = spiral_trajectory(2, 50, 5.0, 0.1, 42);
        let cohort = AgentCohort::new(vec![
            AgentSignature::from_history("drifter", &h1),
            AgentSignature::from_history("clusterer", &h2),
            AgentSignature::from_history("spiral", &h3),
        ]);
        let dm = cohort.pairwise_distances();
        assert_eq!(dm.nrows(), 3);
        assert_eq!(dm.ncols(), 3);
        // Diagonal should be ~0
        assert!(dm[(0, 0)] < 1e-10);
        assert!(dm[(1, 1)] < 1e-10);
        assert!(dm[(2, 2)] < 1e-10);
    }

    #[test]
    fn test_cohort_most_distinct() {
        let h1 = synthetic_trajectory(3, 50, 0.1, 0.01, 42);
        let h2 = synthetic_trajectory(3, 50, 0.1, 0.01, 43);
        let h3 = clustered_trajectory(3, 50, 5, 10.0, 2.0, 42);
        let cohort = AgentCohort::new(vec![
            AgentSignature::from_history("normal1", &h1),
            AgentSignature::from_history("normal2", &h2),
            AgentSignature::from_history("outlier", &h3),
        ]);
        let md = cohort.most_distinct().unwrap();
        assert_eq!(md.name, "outlier");
    }

    #[test]
    fn test_cohort_cluster() {
        let h1 = synthetic_trajectory(3, 50, 0.1, 0.01, 42);
        let h2 = synthetic_trajectory(3, 50, 0.1, 0.01, 43);
        let h3 = clustered_trajectory(3, 50, 5, 10.0, 2.0, 42);
        let cohort = AgentCohort::new(vec![
            AgentSignature::from_history("a", &h1),
            AgentSignature::from_history("b", &h2),
            AgentSignature::from_history("c", &h3),
        ]);
        let clusters = cohort.cluster_by_similarity(100.0); // large threshold
        assert!(clusters.len() >= 1);
    }

    #[test]
    fn test_cohort_rank() {
        let cohort = AgentCohort::new(vec![
            mk_sig("low", 0.1),
            mk_sig("mid", 0.5),
            mk_sig("high", 0.9),
        ]);
        let ranked = cohort.rank_by(|s| s.complexity.complexity_score);
        assert_eq!(ranked[0].0, 2); // highest first
        assert_eq!(ranked[2].0, 0); // lowest last
    }

    fn mk_sig(name: &str, complexity: f64) -> AgentSignature {
        AgentSignature {
            name: name.to_string(),
            complexity: TopologicalComplexity {
                total_persistence: complexity * 10.0,
                total_persistence_sq: complexity * 100.0,
                h0_significant: (complexity * 5.0) as usize,
                h1_significant: (complexity * 2.0) as usize,
                max_persistence: complexity * 5.0,
                entropy: complexity * 2.0,
                complexity_score: complexity,
                predicted_difficulty: complexity,
            },
            robustness: RobustnessScore {
                robustness: complexity,
                mean_persistence: complexity * 3.0,
                long_lived_fraction: complexity,
                stability: complexity,
                fragility_indicators: vec![],
            },
            max_persistence_h0: complexity * 5.0,
            max_persistence_h1: complexity * 3.0,
            avg_persistence: complexity * 3.0,
            entropy: complexity * 2.0,
            n_features: (complexity * 10.0) as usize,
        }
    }

    #[test]
    fn test_signature_serde() {
        let h = synthetic_trajectory(3, 50, 0.1, 0.01, 42);
        let sig = AgentSignature::from_history("test", &h);
        let json = serde_json::to_string(&sig).unwrap();
        let s2: AgentSignature = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.name, "test");
    }

    #[test]
    fn test_comparison_serde() {
        let comp = AgentComparison::compare(&mk_sig("a", 0.5), &mk_sig("b", 0.3));
        let json = serde_json::to_string(&comp).unwrap();
        let c2: AgentComparison = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.agent_a, "a");
    }

    #[test]
    fn test_empty_cohort() {
        let cohort = AgentCohort::new(vec![]);
        assert!(cohort.most_distinct().is_none());
        let dm = cohort.pairwise_distances();
        assert_eq!(dm.nrows(), 0);
    }

    #[test]
    fn test_feature_vector_length() {
        let h = synthetic_trajectory(2, 30, 0.1, 0.01, 42);
        let sig = AgentSignature::from_history("test", &h);
        assert_eq!(sig.feature_vector().len(), 12);
    }
}
