//! Topological complexity as a learning difficulty predictor.

use crate::belief::StateHistory;
use crate::persistence::{compute_persistence, PersistenceDiagram};
use serde::{Deserialize, Serialize};

/// Topological complexity metrics for a belief manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalComplexity {
    /// Total persistence (sum of all persistences).
    pub total_persistence: f64,
    /// Total persistence^2.
    pub total_persistence_sq: f64,
    /// Number of significant H0 features.
    pub h0_significant: usize,
    /// Number of significant H1 features.
    pub h1_significant: usize,
    /// Maximum persistence.
    pub max_persistence: f64,
    /// Persistence entropy.
    pub entropy: f64,
    /// Normalized complexity score (0-1).
    pub complexity_score: f64,
    /// Predicted learning difficulty (0-1).
    pub predicted_difficulty: f64,
}

impl TopologicalComplexity {
    /// Compute from a state history.
    pub fn from_history(history: &StateHistory, significance_threshold: f64) -> Self {
        let dg = compute_persistence(history, 1);
        Self::from_diagram(&dg, significance_threshold)
    }

    /// Compute from a persistence diagram.
    pub fn from_diagram(diagram: &PersistenceDiagram, significance_threshold: f64) -> Self {
        let total_p = diagram.total_persistence(1.0);
        let total_p_sq = diagram.total_persistence(2.0);
        let h0_sig = diagram
            .dimension(0)
            .iter()
            .filter(|p| p.persistence() > significance_threshold && p.is_finite())
            .count();
        let h1_sig = diagram
            .dimension(1)
            .iter()
            .filter(|p| p.persistence() > significance_threshold && p.is_finite())
            .count();
        let max_p = diagram.max_persistence();
        let entropy = diagram.persistence_entropy();

        // Complexity score: combination of total persistence, number of features, entropy
        // Normalized heuristically
        let n_features = diagram.pairs.iter().filter(|p| p.is_finite()).count() as f64;
        let tp = if total_p_sq.is_nan() || total_p_sq.is_infinite() { 0.0 } else { total_p_sq };
        let en = if entropy.is_nan() || entropy.is_infinite() { 0.0 } else { entropy };
        let complexity_score = if tp > 0.0 {
            let p_contrib = (tp / (tp + 1.0)).min(1.0);
            let f_contrib = (n_features / 10.0).min(1.0);
            let e_contrib = (en / 3.0).min(1.0);
            (p_contrib * 0.4 + f_contrib * 0.3 + e_contrib * 0.3).min(1.0)
        } else {
            0.0
        };

        // Predicted difficulty: monotonic function of complexity
        let predicted_difficulty = complexity_score;

        TopologicalComplexity {
            total_persistence: total_p,
            total_persistence_sq: total_p_sq,
            h0_significant: h0_sig,
            h1_significant: h1_sig,
            max_persistence: max_p,
            entropy,
            complexity_score,
            predicted_difficulty,
        }
    }

    /// Is this manifold topologically simple?
    pub fn is_simple(&self) -> bool {
        self.complexity_score < 0.3
    }

    /// Is this manifold topologically complex?
    pub fn is_complex(&self) -> bool {
        self.complexity_score > 0.7
    }

    /// Compare two complexity scores.
    pub fn relative_complexity(&self, other: &TopologicalComplexity) -> f64 {
        if other.total_persistence_sq + self.total_persistence_sq < 1e-15 {
            return 0.0;
        }
        (self.total_persistence_sq - other.total_persistence_sq)
            / (self.total_persistence_sq + other.total_persistence_sq).max(1e-15)
    }
}

/// Track how topological complexity changes over learning steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityTrajectory {
    pub entries: Vec<(usize, TopologicalComplexity)>,
}

impl ComplexityTrajectory {
    pub fn new(entries: Vec<(usize, TopologicalComplexity)>) -> Self {
        Self { entries }
    }

    /// Compute from a history using sliding windows.
    pub fn from_history(
        history: &StateHistory,
        window_size: usize,
        stride: usize,
        significance_threshold: f64,
    ) -> Self {
        let mut entries = Vec::new();
        let mut start = 0;
        while start + window_size <= history.len() {
            let window = StateHistory {
                states: history.states[start..start + window_size].to_vec(),
            };
            let tc = TopologicalComplexity::from_history(&window, significance_threshold);
            entries.push((start, tc));
            start += stride;
        }
        ComplexityTrajectory::new(entries)
    }

    /// Rate of change of complexity.
    pub fn complexity_velocity(&self) -> Vec<(usize, f64)> {
        self.entries
            .windows(2)
            .map(|w| {
                let dt = (w[1].0 - w[0].0) as f64;
                let dc = w[1].1.complexity_score - w[0].1.complexity_score;
                (w[1].0, dc / dt.max(1.0))
            })
            .collect()
    }

    /// Is complexity increasing?
    pub fn is_increasing(&self) -> bool {
        if self.entries.len() < 2 {
            return false;
        }
        let first = self.entries.first().unwrap().1.complexity_score;
        let last = self.entries.last().unwrap().1.complexity_score;
        last > first
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::*;

    #[test]
    fn test_complexity_simple() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0], 0),
            BeliefState::new(vec![0.01], 1),
            BeliefState::new(vec![0.02], 2),
        ]);
        let tc = TopologicalComplexity::from_history(&h, 0.001);
        assert!(tc.complexity_score >= 0.0);
    }

    #[test]
    fn test_complexity_complex() {
        let h = clustered_trajectory(3, 100, 5, 10.0, 1.0, 42);
        let tc = TopologicalComplexity::from_history(&h, 0.5);
        assert!(tc.complexity_score >= 0.0);
    }

    #[test]
    fn test_is_simple() {
        let tc = TopologicalComplexity {
            total_persistence: 0.1,
            total_persistence_sq: 0.01,
            h0_significant: 0,
            h1_significant: 0,
            max_persistence: 0.1,
            entropy: 0.1,
            complexity_score: 0.1,
            predicted_difficulty: 0.1,
        };
        assert!(tc.is_simple());
        assert!(!tc.is_complex());
    }

    #[test]
    fn test_is_complex() {
        let tc = TopologicalComplexity {
            total_persistence: 100.0,
            total_persistence_sq: 1000.0,
            h0_significant: 10,
            h1_significant: 5,
            max_persistence: 50.0,
            entropy: 2.5,
            complexity_score: 0.9,
            predicted_difficulty: 0.9,
        };
        assert!(tc.is_complex());
        assert!(!tc.is_simple());
    }

    #[test]
    fn test_relative_complexity() {
        let tc1 = TopologicalComplexity {
            total_persistence: 10.0,
            total_persistence_sq: 100.0,
            h0_significant: 5,
            h1_significant: 2,
            max_persistence: 5.0,
            entropy: 1.0,
            complexity_score: 0.5,
            predicted_difficulty: 0.5,
        };
        let tc2 = TopologicalComplexity {
            total_persistence: 5.0,
            total_persistence_sq: 25.0,
            h0_significant: 2,
            h1_significant: 1,
            max_persistence: 3.0,
            entropy: 0.5,
            complexity_score: 0.3,
            predicted_difficulty: 0.3,
        };
        let rel = tc1.relative_complexity(&tc2);
        assert!(rel > 0.0); // tc1 is more complex
    }

    #[test]
    fn test_complexity_trajectory() {
        let t = synthetic_trajectory(3, 100, 0.2, 0.1, 42);
        let traj = ComplexityTrajectory::from_history(&t, 20, 10, 0.1);
        assert!(traj.entries.len() > 3);
    }

    #[test]
    fn test_complexity_velocity() {
        let traj = ComplexityTrajectory::new(vec![
            (0, TopologicalComplexity {
                total_persistence: 1.0, total_persistence_sq: 1.0,
                h0_significant: 1, h1_significant: 0,
                max_persistence: 1.0, entropy: 0.5,
                complexity_score: 0.2, predicted_difficulty: 0.2,
            }),
            (10, TopologicalComplexity {
                total_persistence: 2.0, total_persistence_sq: 4.0,
                h0_significant: 2, h1_significant: 1,
                max_persistence: 2.0, entropy: 0.8,
                complexity_score: 0.5, predicted_difficulty: 0.5,
            }),
        ]);
        let vel = traj.complexity_velocity();
        assert_eq!(vel.len(), 1);
        assert!(vel[0].1 > 0.0);
    }

    #[test]
    fn test_is_increasing() {
        let traj = ComplexityTrajectory::new(vec![
            (0, TopologicalComplexity {
                total_persistence: 1.0, total_persistence_sq: 1.0,
                h0_significant: 1, h1_significant: 0,
                max_persistence: 1.0, entropy: 0.5,
                complexity_score: 0.2, predicted_difficulty: 0.2,
            }),
            (10, TopologicalComplexity {
                total_persistence: 5.0, total_persistence_sq: 25.0,
                h0_significant: 3, h1_significant: 2,
                max_persistence: 5.0, entropy: 1.5,
                complexity_score: 0.8, predicted_difficulty: 0.8,
            }),
        ]);
        assert!(traj.is_increasing());
    }

    #[test]
    fn test_is_not_increasing() {
        let traj = ComplexityTrajectory::new(vec![
            (0, TopologicalComplexity {
                total_persistence: 10.0, total_persistence_sq: 100.0,
                h0_significant: 5, h1_significant: 3,
                max_persistence: 10.0, entropy: 2.0,
                complexity_score: 0.9, predicted_difficulty: 0.9,
            }),
            (10, TopologicalComplexity {
                total_persistence: 1.0, total_persistence_sq: 1.0,
                h0_significant: 1, h1_significant: 0,
                max_persistence: 1.0, entropy: 0.3,
                complexity_score: 0.1, predicted_difficulty: 0.1,
            }),
        ]);
        assert!(!traj.is_increasing());
    }

    #[test]
    fn test_complexity_serde() {
        let tc = TopologicalComplexity {
            total_persistence: 5.0,
            total_persistence_sq: 25.0,
            h0_significant: 3,
            h1_significant: 1,
            max_persistence: 3.0,
            entropy: 1.2,
            complexity_score: 0.6,
            predicted_difficulty: 0.6,
        };
        let json = serde_json::to_string(&tc).unwrap();
        let tc2: TopologicalComplexity = serde_json::from_str(&json).unwrap();
        assert!((tc2.total_persistence - 5.0).abs() < 1e-10);
        assert_eq!(tc2.h0_significant, 3);
    }

    #[test]
    fn test_complexity_empty_history() {
        let h = StateHistory::new(vec![]);
        let tc = TopologicalComplexity::from_history(&h, 0.1);
        assert_eq!(tc.complexity_score, 0.0);
    }

    #[test]
    fn test_complexity_trajectory_serde() {
        let traj = ComplexityTrajectory::new(vec![
            (0, TopologicalComplexity {
                total_persistence: 1.0, total_persistence_sq: 1.0,
                h0_significant: 0, h1_significant: 0,
                max_persistence: 0.0, entropy: 0.0,
                complexity_score: 0.0, predicted_difficulty: 0.0,
            }),
        ]);
        let json = serde_json::to_string(&traj).unwrap();
        let traj2: ComplexityTrajectory = serde_json::from_str(&json).unwrap();
        assert_eq!(traj2.entries.len(), 1);
    }
}
