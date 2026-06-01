//! Robustness score: high persistence = robust learner, low persistence = fragile learner.

use crate::belief::StateHistory;
use crate::complexity::TopologicalComplexity;
use crate::persistence::{compute_persistence, PersistenceDiagram};
use serde::{Deserialize, Serialize};

/// Robustness assessment of a learner based on its topological signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessScore {
    /// Overall robustness (0 = fragile, 1 = robust).
    pub robustness: f64,
    /// Mean persistence of features.
    pub mean_persistence: f64,
    /// Fraction of long-lived features.
    pub long_lived_fraction: f64,
    /// Topological stability (consistency of features across windows).
    pub stability: f64,
    /// Fragility indicators.
    pub fragility_indicators: Vec<String>,
}

impl RobustnessScore {
    /// Compute robustness from a state history.
    pub fn from_history(history: &StateHistory, window_size: usize, persistence_threshold: f64) -> Self {
        let window_size = window_size.max(2);
        if history.len() < window_size {
            return RobustnessScore {
                robustness: 0.0,
                mean_persistence: 0.0,
                long_lived_fraction: 0.0,
                stability: 0.0,
                fragility_indicators: vec!["Insufficient data".to_string()],
            };
        }

        // Compute persistence across sliding windows
        let mut diagrams = Vec::new();
        let mut start = 0;
        let stride = (window_size / 2).max(1);
        while start + window_size <= history.len() {
            let window = StateHistory {
                states: history.states[start..start + window_size].to_vec(),
            };
            let dg = compute_persistence(&window, 1);
            diagrams.push(dg);
            start += stride;
        }

        Self::from_diagrams(&diagrams, persistence_threshold)
    }

    /// Compute robustness from multiple persistence diagrams.
    pub fn from_diagrams(diagrams: &[PersistenceDiagram], persistence_threshold: f64) -> Self {
        let mut fragility_indicators = Vec::new();

        // Mean persistence across all diagrams
        let mean_persistence: f64 = if diagrams.is_empty() {
            0.0
        } else {
            let total: f64 = diagrams.iter().map(|d| d.avg_persistence()).sum();
            let avg = total / diagrams.len() as f64;
            if avg.is_nan() || avg.is_infinite() { 0.0 } else { avg }
        };

        // Long-lived fraction
        let total_features: f64 = diagrams
            .iter()
            .map(|d| d.pairs.iter().filter(|p| p.is_finite()).count() as f64)
            .sum();
        let long_lived: f64 = diagrams
            .iter()
            .map(|d| {
                d.pairs
                    .iter()
                    .filter(|p| p.is_finite() && p.persistence() > persistence_threshold)
                    .count() as f64
            })
            .sum();
        let long_lived_fraction = if total_features > 0.0 {
            long_lived / total_features
        } else {
            0.0
        };

        // Stability: how consistent are the top features across windows
        let stability = if diagrams.len() < 2 {
            1.0
        } else {
            let max_persistences: Vec<f64> = diagrams.iter().map(|d| d.max_persistence()).collect();
            let mean_mp = max_persistences.iter().sum::<f64>() / max_persistences.len() as f64;
            if mean_mp < 1e-10 {
                1.0
            } else {
                let variance = max_persistences
                    .iter()
                    .map(|&v| (v - mean_mp).powi(2))
                    .sum::<f64>()
                    / max_persistences.len() as f64;
                1.0 / (1.0 + variance / (mean_mp * mean_mp))
            }
        };

        // Overall robustness: weighted combination
        let persistence_norm = if mean_persistence.is_nan() || mean_persistence.is_infinite() {
            0.0
        } else {
            (mean_persistence.clamp(0.0, 5.0) / 5.0)
        };
        let robustness = persistence_norm * 0.4
            + long_lived_fraction * 0.3
            + stability * 0.3;

        // Fragility indicators
        if mean_persistence < 0.5 {
            fragility_indicators.push("Very low mean persistence — features barely survive".to_string());
        }
        if long_lived_fraction < 0.2 {
            fragility_indicators.push("Few long-lived features — topological structure is transient".to_string());
        }
        if stability < 0.3 {
            fragility_indicators.push("Low stability — topological features inconsistent across windows".to_string());
        }
        if total_features < 3.0 {
            fragility_indicators.push("Very few topological features detected".to_string());
        }

        RobustnessScore {
            robustness: robustness.clamp(0.0, 1.0),
            mean_persistence,
            long_lived_fraction,
            stability,
            fragility_indicators,
        }
    }

    /// Is this a robust learner?
    pub fn is_robust(&self) -> bool {
        self.robustness > 0.6
    }

    /// Is this a fragile learner?
    pub fn is_fragile(&self) -> bool {
        self.robustness < 0.3
    }

    /// Robustness category.
    pub fn category(&self) -> &'static str {
        if self.robustness > 0.8 {
            "Highly Robust"
        } else if self.robustness > 0.6 {
            "Robust"
        } else if self.robustness > 0.4 {
            "Moderate"
        } else if self.robustness > 0.2 {
            "Fragile"
        } else {
            "Highly Fragile"
        }
    }
}

/// Compare robustness between two learners.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessComparison {
    pub score_a: RobustnessScore,
    pub score_b: RobustnessScore,
    pub robustness_delta: f64,
    pub prediction: RobustnessPrediction,
}

/// Prediction about relative learning behavior.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RobustnessPrediction {
    /// A will learn slower but be more robust.
    ASlowerRobust,
    /// B will learn slower but be more robust.
    BSlowerRobust,
    /// Both similar.
    Similar,
}

impl RobustnessComparison {
    pub fn new(score_a: RobustnessScore, score_b: RobustnessScore) -> Self {
        let delta = score_a.robustness - score_b.robustness;
        let prediction = if delta.abs() < 0.1 {
            RobustnessPrediction::Similar
        } else if delta > 0.0 {
            RobustnessPrediction::ASlowerRobust
        } else {
            RobustnessPrediction::BSlowerRobust
        };
        RobustnessComparison {
            score_a,
            score_b,
            robustness_delta: delta,
            prediction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::*;

    #[test]
    fn test_robustness_from_simple_history() {
        let h = StateHistory::new(vec![
            BeliefState::new(vec![0.0], 0),
            BeliefState::new(vec![1.0], 1),
            BeliefState::new(vec![2.0], 2),
        ]);
        let score = RobustnessScore::from_history(&h, 2, 0.5);
        assert!(score.robustness >= 0.0);
        assert!(score.robustness <= 1.0);
    }

    #[test]
    fn test_robustness_insufficient_data() {
        let h = StateHistory::new(vec![BeliefState::new(vec![0.0], 0)]);
        let score = RobustnessScore::from_history(&h, 10, 0.5);
        assert_eq!(score.robustness, 0.0);
        assert!(score.fragility_indicators.len() > 0);
    }

    #[test]
    fn test_robustness_categories() {
        let robust = mk_score(0.9);
        assert!(robust.is_robust());
        assert!(!robust.is_fragile());
        assert_eq!(robust.category(), "Highly Robust");

        let fragile = mk_score(0.1);
        assert!(fragile.is_fragile());
        assert_eq!(fragile.category(), "Highly Fragile");

        let moderate = mk_score(0.5);
        assert_eq!(moderate.category(), "Moderate");
    }

    fn mk_score(r: f64) -> RobustnessScore {
        RobustnessScore {
            robustness: r,
            mean_persistence: r * 3.0,
            long_lived_fraction: r,
            stability: r,
            fragility_indicators: vec![],
        }
    }

    #[test]
    fn test_robustness_from_diagrams() {
        let dg = PersistenceDiagram::new(
            vec![
                crate::persistence::PersistencePair::new(0.0, 5.0, 0),
                crate::persistence::PersistencePair::new(0.0, 3.0, 0),
            ],
            5,
        );
        let score = RobustnessScore::from_diagrams(&[dg], 1.0);
        assert!(score.mean_persistence > 0.0);
        assert!(score.long_lived_fraction > 0.0);
    }

    #[test]
    fn test_robustness_comparison_similar() {
        let comp = RobustnessComparison::new(mk_score(0.5), mk_score(0.52));
        assert_eq!(comp.prediction, RobustnessPrediction::Similar);
    }

    #[test]
    fn test_robustness_comparison_a_robust() {
        let comp = RobustnessComparison::new(mk_score(0.8), mk_score(0.2));
        assert_eq!(comp.prediction, RobustnessPrediction::ASlowerRobust);
        assert!(comp.robustness_delta > 0.0);
    }

    #[test]
    fn test_robustness_comparison_b_robust() {
        let comp = RobustnessComparison::new(mk_score(0.2), mk_score(0.8));
        assert_eq!(comp.prediction, RobustnessPrediction::BSlowerRobust);
        assert!(comp.robustness_delta < 0.0);
    }

    #[test]
    fn test_robustness_spiral() {
        let t = spiral_trajectory(2, 100, 5.0, 0.1, 42);
        let score = RobustnessScore::from_history(&t, 20, 0.5);
        assert!(score.robustness >= 0.0);
    }

    #[test]
    fn test_robustness_clustered() {
        let t = clustered_trajectory(3, 100, 4, 5.0, 0.5, 42);
        let score = RobustnessScore::from_history(&t, 20, 0.5);
        assert!(score.robustness >= 0.0);
    }

    #[test]
    fn test_fragility_indicators_low_persistence() {
        let dg = PersistenceDiagram::new(
            vec![crate::persistence::PersistencePair::new(0.0, 0.1, 0)],
            3,
        );
        let score = RobustnessScore::from_diagrams(&[dg], 1.0);
        assert!(score.fragility_indicators.len() > 0);
    }

    #[test]
    fn test_robustness_score_serde() {
        let score = mk_score(0.7);
        let json = serde_json::to_string(&score).unwrap();
        let s2: RobustnessScore = serde_json::from_str(&json).unwrap();
        assert!((s2.robustness - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_comparison_serde() {
        let comp = RobustnessComparison::new(mk_score(0.8), mk_score(0.3));
        let json = serde_json::to_string(&comp).unwrap();
        let c2: RobustnessComparison = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.prediction, RobustnessPrediction::ASlowerRobust);
    }

    #[test]
    fn test_empty_diagrams_robustness() {
        let score = RobustnessScore::from_diagrams(&[], 0.5);
        assert_eq!(score.mean_persistence, 0.0);
    }
}
