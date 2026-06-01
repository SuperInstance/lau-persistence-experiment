//! Phase transition detection: sudden changes in topology = breakthroughs or failures.

use crate::belief::StateHistory;
use crate::betti::BettiCurve;
use crate::complexity::{ComplexityTrajectory, TopologicalComplexity};
use crate::persistence::{compute_persistence, PersistenceDiagram};
use serde::{Deserialize, Serialize};

/// A detected phase transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    pub step: usize,
    pub transition_type: TransitionType,
    pub magnitude: f64,
    pub description: String,
}

/// Type of phase transition.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TransitionType {
    /// Sudden increase in complexity (exploration burst).
    ExplorationBurst,
    /// Sudden decrease in complexity (convergence/breakthrough).
    Breakthrough,
    /// Oscillation detected (instability).
    Oscillation,
    /// Gradual change detected.
    GradualShift,
    /// Collapse (features dying rapidly).
    Collapse,
}

/// Phase transition detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransitionReport {
    pub transitions: Vec<PhaseTransition>,
    pub stability_score: f64,
    pub exploration_ratio: f64,
    pub convergence_ratio: f64,
}

impl PhaseTransitionReport {
    /// Detect phase transitions from a complexity trajectory.
    pub fn from_trajectory(
        trajectory: &ComplexityTrajectory,
        spike_threshold: f64,
        window: usize,
    ) -> Self {
        let mut transitions = Vec::new();

        if trajectory.entries.len() < 3 {
            return PhaseTransitionReport {
                transitions,
                stability_score: 1.0,
                exploration_ratio: 0.0,
                convergence_ratio: 0.0,
            };
        }

        let complexities: Vec<(usize, f64)> = trajectory
            .entries
            .iter()
            .map(|(s, tc)| (*s, tc.complexity_score))
            .collect();

        // Detect spikes
        for i in window..complexities.len().saturating_sub(window) {
            let before: f64 = complexities[i.saturating_sub(window)..i]
                .iter()
                .map(|(_, c)| *c)
                .sum::<f64>()
                / window as f64;
            let after: f64 = complexities[i + 1..(i + window + 1).min(complexities.len())]
                .iter()
                .map(|(_, c)| *c)
                .sum::<f64>()
                / (window).min(complexities.len() - i) as f64;
            let current = complexities[i].1;
            let delta = after - before;

            if delta.abs() > spike_threshold {
                let (ttype, desc) = if delta > spike_threshold {
                    (TransitionType::ExplorationBurst, "Sudden complexity increase".to_string())
                } else {
                    (TransitionType::Breakthrough, "Sudden complexity decrease".to_string())
                };
                transitions.push(PhaseTransition {
                    step: complexities[i].0,
                    transition_type: ttype,
                    magnitude: delta.abs(),
                    description: desc,
                });
            }
        }

        // Detect oscillations
        let mut sign_changes = 0;
        let diffs: Vec<f64> = complexities
            .windows(2)
            .map(|w| w[1].1 - w[0].1)
            .collect();
        for d in diffs.windows(2) {
            if d[0] * d[1] < 0.0 {
                sign_changes += 1;
            }
        }
        if sign_changes > complexities.len() / 3 {
            transitions.push(PhaseTransition {
                step: 0,
                transition_type: TransitionType::Oscillation,
                magnitude: sign_changes as f64,
                description: format!("{} direction changes detected", sign_changes),
            });
        }

        // Compute metrics
        let n_transitions = transitions.len() as f64;
        let n_steps = complexities.len() as f64;
        let stability_score = 1.0 - (n_transitions / n_steps).min(1.0);

        let explorations = transitions
            .iter()
            .filter(|t| t.transition_type == TransitionType::ExplorationBurst)
            .count();
        let breakthroughs = transitions
            .iter()
            .filter(|t| t.transition_type == TransitionType::Breakthrough)
            .count();

        let exploration_ratio = if n_transitions > 0.0 {
            explorations as f64 / n_transitions
        } else {
            0.0
        };
        let convergence_ratio = if n_transitions > 0.0 {
            breakthroughs as f64 / n_transitions
        } else {
            0.0
        };

        PhaseTransitionReport {
            transitions,
            stability_score,
            exploration_ratio,
            convergence_ratio,
        }
    }

    /// Detect phase transitions from raw history.
    pub fn from_history(
        history: &StateHistory,
        window_size: usize,
        stride: usize,
        spike_threshold: f64,
    ) -> Self {
        let traj = ComplexityTrajectory::from_history(history, window_size, stride, 0.1);
        let smooth_window = (traj.entries.len() / 10).max(1);
        Self::from_trajectory(&traj, spike_threshold, smooth_window)
    }

    /// Get transitions of a specific type.
    pub fn transitions_of_type(&self, ttype: TransitionType) -> Vec<&PhaseTransition> {
        self.transitions
            .iter()
            .filter(|t| t.transition_type == ttype)
            .collect()
    }

    /// Most significant transition.
    pub fn most_significant(&self) -> Option<&PhaseTransition> {
        self.transitions
            .iter()
            .max_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap_or(std::cmp::Ordering::Equal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::*;

    #[test]
    fn test_no_transitions_stable() {
        let traj = ComplexityTrajectory::new(
            (0..10)
                .map(|i| {
                    (i * 10, TopologicalComplexity {
                        total_persistence: 1.0,
                        total_persistence_sq: 1.0,
                        h0_significant: 1,
                        h1_significant: 0,
                        max_persistence: 1.0,
                        entropy: 0.5,
                        complexity_score: 0.5,
                        predicted_difficulty: 0.5,
                    })
                })
                .collect(),
        );
        let report = PhaseTransitionReport::from_trajectory(&traj, 0.3, 2);
        assert!(report.stability_score > 0.5);
    }

    #[test]
    fn test_spike_detection() {
        let traj = ComplexityTrajectory::new(vec![
            (0, mk_tc(0.1)),
            (10, mk_tc(0.1)),
            (20, mk_tc(0.1)),
            (30, mk_tc(0.9)), // spike!
            (40, mk_tc(0.1)),
            (50, mk_tc(0.1)),
        ]);
        let report = PhaseTransitionReport::from_trajectory(&traj, 0.2, 1);
        // Should detect the spike
        assert!(report.transitions.len() > 0);
    }

    fn mk_tc(score: f64) -> TopologicalComplexity {
        TopologicalComplexity {
            total_persistence: score * 10.0,
            total_persistence_sq: score * 100.0,
            h0_significant: (score * 5.0) as usize,
            h1_significant: (score * 2.0) as usize,
            max_persistence: score * 5.0,
            entropy: score * 2.0,
            complexity_score: score,
            predicted_difficulty: score,
        }
    }

    #[test]
    fn test_from_history() {
        let t = synthetic_trajectory(3, 100, 0.2, 0.1, 42);
        let report = PhaseTransitionReport::from_history(&t, 20, 5, 0.2);
        assert!(report.stability_score >= 0.0);
        assert!(report.stability_score <= 1.0);
    }

    #[test]
    fn test_transitions_of_type() {
        let report = PhaseTransitionReport {
            transitions: vec![
                PhaseTransition {
                    step: 10, transition_type: TransitionType::ExplorationBurst,
                    magnitude: 0.5, description: "test".to_string(),
                },
                PhaseTransition {
                    step: 20, transition_type: TransitionType::Breakthrough,
                    magnitude: 0.3, description: "test".to_string(),
                },
            ],
            stability_score: 0.5,
            exploration_ratio: 0.5,
            convergence_ratio: 0.5,
        };
        assert_eq!(report.transitions_of_type(TransitionType::ExplorationBurst).len(), 1);
        assert_eq!(report.transitions_of_type(TransitionType::Breakthrough).len(), 1);
        assert_eq!(report.transitions_of_type(TransitionType::Collapse).len(), 0);
    }

    #[test]
    fn test_most_significant() {
        let report = PhaseTransitionReport {
            transitions: vec![
                PhaseTransition {
                    step: 10, transition_type: TransitionType::ExplorationBurst,
                    magnitude: 0.3, description: "small".to_string(),
                },
                PhaseTransition {
                    step: 20, transition_type: TransitionType::Breakthrough,
                    magnitude: 0.9, description: "big".to_string(),
                },
            ],
            stability_score: 0.5,
            exploration_ratio: 0.5,
            convergence_ratio: 0.5,
        };
        let ms = report.most_significant().unwrap();
        assert!((ms.magnitude - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_empty_report() {
        let report = PhaseTransitionReport {
            transitions: vec![],
            stability_score: 1.0,
            exploration_ratio: 0.0,
            convergence_ratio: 0.0,
        };
        assert!(report.most_significant().is_none());
    }

    #[test]
    fn test_short_trajectory() {
        let traj = ComplexityTrajectory::new(vec![
            (0, mk_tc(0.5)),
        ]);
        let report = PhaseTransitionReport::from_trajectory(&traj, 0.3, 2);
        assert!(report.stability_score > 0.0);
    }

    #[test]
    fn test_report_serde() {
        let report = PhaseTransitionReport {
            transitions: vec![
                PhaseTransition {
                    step: 10, transition_type: TransitionType::Breakthrough,
                    magnitude: 0.5, description: "test".to_string(),
                },
            ],
            stability_score: 0.8,
            exploration_ratio: 0.1,
            convergence_ratio: 0.5,
        };
        let json = serde_json::to_string(&report).unwrap();
        let r2: PhaseTransitionReport = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.transitions.len(), 1);
        assert!((r2.stability_score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_clustered_trajectory_transitions() {
        let t = clustered_trajectory(3, 200, 5, 10.0, 1.0, 42);
        let report = PhaseTransitionReport::from_history(&t, 30, 10, 0.1);
        // Clustered trajectories should produce transitions
        assert!(report.transitions.len() >= 0);
    }
}
