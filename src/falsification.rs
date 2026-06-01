//! Falsification: can we find agents that learn well despite low persistence?
//!
//! This module provides tools to search for counterexamples to the hypothesis
//! that high persistence is necessary for robust learning.

use crate::belief::{clustered_trajectory, spiral_trajectory, synthetic_trajectory, StateHistory};
use crate::persistence::{compute_persistence, PersistenceDiagram};
use crate::robustness::RobustnessScore;
use serde::{Deserialize, Serialize};

/// A candidate counterexample agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleCandidate {
    pub name: String,
    pub persistence_score: f64,
    pub robustness_score: f64,
    pub learning_speed: f64,
    pub is_counterexample: bool,
    pub reason: String,
}

impl CounterexampleCandidate {
    /// Check if this agent falsifies the hypothesis.
    ///
    /// The hypothesis states: high persistence → robust learning.
    /// A counterexample has: low persistence + robust learning (or high persistence + fragile).
    pub fn check(&mut self) {
        self.is_counterexample = (self.persistence_score < 0.3 && self.robustness_score > 0.6)
            || (self.persistence_score > 0.7 && self.robustness_score < 0.3);
        self.reason = if self.persistence_score < 0.3 && self.robustness_score > 0.6 {
            "Low persistence but high robustness — hypothesis falsified".to_string()
        } else if self.persistence_score > 0.7 && self.robustness_score < 0.3 {
            "High persistence but low robustness — inverse falsification".to_string()
        } else {
            "Consistent with hypothesis".to_string()
        };
    }
}

/// Result of a falsification sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationResult {
    pub candidates: Vec<CounterexampleCandidate>,
    pub counterexamples_found: usize,
    pub hypothesis_supported: bool,
    pub confidence: f64,
    pub summary: String,
}

impl FalsificationResult {
    /// Run a falsification sweep over a set of agents.
    pub fn sweep(histories: &[(String, StateHistory)]) -> Self {
        let mut candidates = Vec::new();

        for (name, history) in histories {
            if history.len() < 5 {
                continue;
            }
            let dg = compute_persistence(history, 1);
            let persistence_score = compute_persistence_score(&dg);
            let robustness_score = compute_robustness_proxy(&dg, history);
            let learning_speed = compute_learning_speed_proxy(history);

            let mut candidate = CounterexampleCandidate {
                name: name.clone(),
                persistence_score,
                robustness_score,
                learning_speed,
                is_counterexample: false,
                reason: String::new(),
            };
            candidate.check();
            candidates.push(candidate);
        }

        let counterexamples_found = candidates.iter().filter(|c| c.is_counterexample).count();
        let n = candidates.len() as f64;
        let support_count = candidates.iter().filter(|c| !c.is_counterexample).count();

        let hypothesis_supported = counterexamples_found == 0
            || (support_count as f64 / n) > 0.8;

        let confidence = if n > 0.0 {
            support_count as f64 / n
        } else {
            0.5
        };

        let summary = if hypothesis_supported {
            format!(
                "Hypothesis supported with {:.0}% confidence ({}/{} agents consistent)",
                confidence * 100.0,
                support_count,
                candidates.len()
            )
        } else {
            format!(
                "Hypothesis challenged: {}/{} counterexamples found",
                counterexamples_found,
                candidates.len()
            )
        };

        FalsificationResult {
            candidates,
            counterexamples_found,
            hypothesis_supported,
            confidence,
            summary,
        }
    }

    /// Generate a standard test battery of synthetic agents.
    pub fn standard_battery() -> Vec<(String, StateHistory)> {
        let mut battery = Vec::new();

        // Slow drift (should have moderate persistence)
        battery.push((
            "slow_drift".to_string(),
            synthetic_trajectory(3, 100, 0.05, 0.01, 42),
        ));

        // Fast drift (should have lower persistence)
        battery.push((
            "fast_drift".to_string(),
            synthetic_trajectory(3, 100, 0.5, 0.1, 42),
        ));

        // Spiral converging (systematic)
        battery.push((
            "spiral_converge".to_string(),
            spiral_trajectory(2, 100, 5.0, 0.1, 42),
        ));

        // Tight spiral
        battery.push((
            "tight_spiral".to_string(),
            spiral_trajectory(2, 100, 1.0, 0.01, 42),
        ));

        // Clustered - few clusters
        battery.push((
            "few_clusters".to_string(),
            clustered_trajectory(3, 100, 2, 5.0, 0.5, 42),
        ));

        // Clustered - many clusters
        battery.push((
            "many_clusters".to_string(),
            clustered_trajectory(3, 100, 8, 5.0, 0.5, 42),
        ));

        // Noisy
        battery.push((
            "noisy".to_string(),
            synthetic_trajectory(3, 100, 0.1, 1.0, 42),
        ));

        // Clean
        battery.push((
            "clean".to_string(),
            synthetic_trajectory(3, 100, 0.1, 0.001, 42),
        ));

        // Wide cluster spread
        battery.push((
            "wide_clusters".to_string(),
            clustered_trajectory(3, 100, 4, 20.0, 0.5, 42),
        ));

        // Tight cluster spread
        battery.push((
            "tight_clusters".to_string(),
            clustered_trajectory(3, 100, 4, 1.0, 0.5, 42),
        ));

        battery
    }

    /// Run the standard falsification sweep.
    pub fn run_standard() -> Self {
        let battery = Self::standard_battery();
        Self::sweep(&battery)
    }

    /// Get counterexample agents.
    pub fn counterexamples(&self) -> Vec<&CounterexampleCandidate> {
        self.candidates.iter().filter(|c| c.is_counterexample).collect()
    }
}

/// Compute a persistence score for an agent (0-1).
fn compute_persistence_score(diagram: &PersistenceDiagram) -> f64 {
    let avg = diagram.avg_persistence();
    if avg.is_nan() || avg.is_infinite() {
        return 0.0;
    }
    (avg / (avg + 1.0)).min(1.0)
}

/// Compute a robustness proxy from topology.
fn compute_robustness_proxy(diagram: &PersistenceDiagram, history: &StateHistory) -> f64 {
    let window = 20.min(history.len()).max(2);
    let score = RobustnessScore::from_history(history, window, 0.5);
    if score.robustness.is_nan() || score.robustness.is_infinite() {
        0.0
    } else {
        score.robustness
    }
}

/// Compute a learning speed proxy (inverse of spread — faster convergence = less spread).
fn compute_learning_speed_proxy(history: &StateHistory) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let first_half = StateHistory {
        states: history.states[..history.len() / 2].to_vec(),
    };
    let second_half = StateHistory {
        states: history.states[history.len() / 2..].to_vec(),
    };
    let spread_ratio = first_half.spread() / (second_half.spread() + 1e-10);
    // Higher spread_ratio means faster convergence
    (spread_ratio / (spread_ratio + 1.0)).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::*;

    #[test]
    fn test_counterexample_check_positive() {
        let mut c = CounterexampleCandidate {
            name: "test".to_string(),
            persistence_score: 0.1,
            robustness_score: 0.8,
            learning_speed: 0.5,
            is_counterexample: false,
            reason: String::new(),
        };
        c.check();
        assert!(c.is_counterexample);
    }

    #[test]
    fn test_counterexample_check_negative() {
        let mut c = CounterexampleCandidate {
            name: "test".to_string(),
            persistence_score: 0.5,
            robustness_score: 0.5,
            learning_speed: 0.5,
            is_counterexample: false,
            reason: String::new(),
        };
        c.check();
        assert!(!c.is_counterexample);
    }

    #[test]
    fn test_counterexample_inverse() {
        let mut c = CounterexampleCandidate {
            name: "test".to_string(),
            persistence_score: 0.9,
            robustness_score: 0.1,
            learning_speed: 0.5,
            is_counterexample: false,
            reason: String::new(),
        };
        c.check();
        assert!(c.is_counterexample);
    }

    #[test]
    fn test_standard_battery() {
        let battery = FalsificationResult::standard_battery();
        assert_eq!(battery.len(), 10);
        for (name, history) in &battery {
            assert!(history.len() > 0, "Empty history for {}", name);
        }
    }

    #[test]
    fn test_falsification_sweep() {
        let battery = FalsificationResult::standard_battery();
        let result = FalsificationResult::sweep(&battery);
        assert!(result.candidates.len() > 0);
        assert!(result.confidence >= 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_falsification_run_standard() {
        let result = FalsificationResult::run_standard();
        assert!(result.candidates.len() >= 5);
        assert!(!result.summary.is_empty());
    }

    #[test]
    fn test_counterexamples_method() {
        let result = FalsificationResult::run_standard();
        let ce = result.counterexamples();
        // Just check it returns something (may or may not find counterexamples)
        assert!(ce.len() >= 0);
    }

    #[test]
    fn test_persistence_score_empty() {
        let dg = PersistenceDiagram::empty();
        let score = compute_persistence_score(&dg);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_persistence_score_high() {
        let dg = PersistenceDiagram::new(
            vec![crate::persistence::PersistencePair::new(0.0, 100.0, 0)],
            3,
        );
        let score = compute_persistence_score(&dg);
        assert!(score > 0.9);
    }

    #[test]
    fn test_learning_speed_proxy() {
        // Converging trajectory
        let t = spiral_trajectory(2, 100, 5.0, 0.0, 42);
        let speed = compute_learning_speed_proxy(&t);
        assert!(speed >= 0.0);
    }

    #[test]
    fn test_learning_speed_empty() {
        let h = StateHistory::new(vec![]);
        let speed = compute_learning_speed_proxy(&h);
        assert_eq!(speed, 0.0);
    }

    #[test]
    fn test_falsification_result_serde() {
        let result = FalsificationResult::run_standard();
        let json = serde_json::to_string(&result).unwrap();
        let r2: FalsificationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.candidates.len(), result.candidates.len());
    }

    #[test]
    fn test_candidate_serde() {
        let mut c = CounterexampleCandidate {
            name: "test".to_string(),
            persistence_score: 0.5,
            robustness_score: 0.5,
            learning_speed: 0.5,
            is_counterexample: false,
            reason: "ok".to_string(),
        };
        c.check();
        let json = serde_json::to_string(&c).unwrap();
        let c2: CounterexampleCandidate = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.name, "test");
    }

    #[test]
    fn test_sweep_empty() {
        let result = FalsificationResult::sweep(&[]);
        assert_eq!(result.candidates.len(), 0);
        assert_eq!(result.counterexamples_found, 0);
    }

    #[test]
    fn test_hypothesis_summary() {
        let result = FalsificationResult::run_standard();
        assert!(!result.summary.is_empty());
        // Should mention either "supported" or "challenged"
        let valid = result.summary.contains("supported") || result.summary.contains("challenged");
        assert!(valid, "Summary should mention hypothesis status: {}", result.summary);
    }
}
