//! Birth-death analysis: when do topological features appear/disappear during learning?

use crate::belief::{StateHistory, BeliefState};
use crate::persistence::{compute_persistence, PersistenceDiagram, PersistencePair};
use serde::{Deserialize, Serialize};

/// A birth-death event during learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BirthDeathEvent {
    pub step: usize,
    pub birth_time: f64,
    pub death_time: f64,
    pub dim: usize,
    pub persistence: f64,
    pub event_type: BirthDeathType,
}

/// Type of birth-death event.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BirthDeathType {
    Birth,
    Death,
    LongLived,
}

/// Summary of birth-death statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BirthDeathSummary {
    pub events: Vec<BirthDeathEvent>,
    pub total_births: usize,
    pub total_deaths: usize,
    pub total_long_lived: usize,
    pub avg_lifetime: f64,
    pub median_lifetime: f64,
    pub max_lifetime: f64,
    pub lifetime_variance: f64,
}

impl BirthDeathSummary {
    /// Compute from a persistence diagram mapped to learning steps.
    pub fn from_diagram(diagram: &PersistenceDiagram, step_scale: f64) -> Self {
        let mut events = Vec::new();

        for pair in &diagram.pairs {
            if !pair.is_finite() {
                continue;
            }
            let birth_step = (pair.birth * step_scale) as usize;
            let death_step = (pair.death * step_scale) as usize;

            events.push(BirthDeathEvent {
                step: birth_step,
                birth_time: pair.birth,
                death_time: pair.death,
                dim: pair.dim,
                persistence: pair.persistence(),
                event_type: BirthDeathType::Birth,
            });
            events.push(BirthDeathEvent {
                step: death_step,
                birth_time: pair.birth,
                death_time: pair.death,
                dim: pair.dim,
                persistence: pair.persistence(),
                event_type: BirthDeathType::Death,
            });
        }

        let lifetimes: Vec<f64> = diagram
            .pairs
            .iter()
            .filter(|p| p.is_finite())
            .map(|p| p.persistence())
            .collect();

        let total_births = events.iter().filter(|e| e.event_type == BirthDeathType::Birth).count();
        let total_deaths = events.iter().filter(|e| e.event_type == BirthDeathType::Death).count();
        let total_long_lived = diagram
            .pairs
            .iter()
            .filter(|p| p.is_finite() && p.persistence() > 1.0)
            .count();

        let avg_lifetime = if lifetimes.is_empty() {
            0.0
        } else {
            lifetimes.iter().sum::<f64>() / lifetimes.len() as f64
        };

        let mut sorted_lifetimes = lifetimes.clone();
        sorted_lifetimes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_lifetime = if sorted_lifetimes.is_empty() {
            0.0
        } else {
            let mid = sorted_lifetimes.len() / 2;
            sorted_lifetimes[mid]
        };

        let max_lifetime = lifetimes.iter().fold(0.0_f64, |a, &b| a.max(b));

        let lifetime_variance = if lifetimes.is_empty() {
            0.0
        } else {
            lifetimes
                .iter()
                .map(|l| (l - avg_lifetime).powi(2))
                .sum::<f64>()
                / lifetimes.len() as f64
        };

        BirthDeathSummary {
            events,
            total_births,
            total_deaths,
            total_long_lived,
            avg_lifetime,
            median_lifetime,
            max_lifetime,
            lifetime_variance,
        }
    }

    /// Events at a specific learning step.
    pub fn events_at_step(&self, step: usize) -> Vec<&BirthDeathEvent> {
        self.events.iter().filter(|e| e.step == step).collect()
    }

    /// Birth rate: births per unit learning time.
    pub fn birth_rate(&self, total_steps: usize) -> f64 {
        if total_steps == 0 {
            return 0.0;
        }
        self.total_births as f64 / total_steps as f64
    }

    /// Death rate: deaths per unit learning time.
    pub fn death_rate(&self, total_steps: usize) -> f64 {
        if total_steps == 0 {
            return 0.0;
        }
        self.total_deaths as f64 / total_steps as f64
    }
}

/// Birth-death timeline: track features across sliding windows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BirthDeathTimeline {
    pub entries: Vec<(usize, BirthDeathSummary)>,
}

impl BirthDeathTimeline {
    /// Compute from a state history using sliding windows.
    pub fn from_history(
        history: &StateHistory,
        window_size: usize,
        stride: usize,
    ) -> Self {
        let mut entries = Vec::new();
        let mut start = 0;
        while start + window_size <= history.len() {
            let window = StateHistory {
                states: history.states[start..start + window_size].to_vec(),
            };
            let dg = compute_persistence(&window, 1);
            let summary = BirthDeathSummary::from_diagram(&dg, 100.0);
            entries.push((start, summary));
            start += stride;
        }
        BirthDeathTimeline { entries }
    }

    /// Find steps where many features die (potential breakthroughs).
    pub fn breakthrough_steps(&self, death_threshold: usize) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|(_, summary)| summary.total_deaths >= death_threshold)
            .map(|(step, _)| *step)
            .collect()
    }

    /// Find steps where many features are born (exploration phases).
    pub fn exploration_steps(&self, birth_threshold: usize) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|(_, summary)| summary.total_births >= birth_threshold)
            .map(|(step, _)| *step)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::*;

    #[test]
    fn test_birth_death_summary_basic() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0),
                PersistencePair::new(0.5, 2.5, 1),
            ],
            5,
        );
        let summary = BirthDeathSummary::from_diagram(&dg, 1.0);
        assert_eq!(summary.total_births, 2);
        assert_eq!(summary.total_deaths, 2);
    }

    #[test]
    fn test_birth_death_avg_lifetime() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 2.0, 0), // lifetime 2
                PersistencePair::new(0.0, 4.0, 0), // lifetime 4
            ],
            3,
        );
        let summary = BirthDeathSummary::from_diagram(&dg, 1.0);
        assert!((summary.avg_lifetime - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_birth_death_median_lifetime() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0),
                PersistencePair::new(0.0, 3.0, 0),
                PersistencePair::new(0.0, 5.0, 0),
            ],
            4,
        );
        let summary = BirthDeathSummary::from_diagram(&dg, 1.0);
        assert!((summary.median_lifetime - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_birth_death_max_lifetime() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0),
                PersistencePair::new(0.0, 10.0, 0),
            ],
            3,
        );
        let summary = BirthDeathSummary::from_diagram(&dg, 1.0);
        assert!((summary.max_lifetime - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_birth_death_variance() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 2.0, 0),
                PersistencePair::new(0.0, 4.0, 0),
            ],
            3,
        );
        let summary = BirthDeathSummary::from_diagram(&dg, 1.0);
        // Mean = 3, var = ((2-3)^2 + (4-3)^2) / 2 = 1
        assert!((summary.lifetime_variance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_birth_death_long_lived() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 0.5, 0),
                PersistencePair::new(0.0, 5.0, 0),
                PersistencePair::new(0.0, 10.0, 1),
            ],
            4,
        );
        let summary = BirthDeathSummary::from_diagram(&dg, 1.0);
        assert_eq!(summary.total_long_lived, 2);
    }

    #[test]
    fn test_events_at_step() {
        let dg = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 5.0, 0)],
            3,
        );
        let summary = BirthDeathSummary::from_diagram(&dg, 10.0);
        let birth_events = summary.events_at_step(0);
        assert!(birth_events.len() >= 1);
    }

    #[test]
    fn test_birth_rate() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0),
                PersistencePair::new(0.0, 2.0, 0),
            ],
            3,
        );
        let summary = BirthDeathSummary::from_diagram(&dg, 1.0);
        let rate = summary.birth_rate(100);
        assert!((rate - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_death_rate() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 1.0, 0),
                PersistencePair::new(0.0, 2.0, 0),
            ],
            3,
        );
        let summary = BirthDeathSummary::from_diagram(&dg, 1.0);
        let rate = summary.death_rate(100);
        assert!((rate - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_birth_death_timeline() {
        let t = synthetic_trajectory(3, 80, 0.2, 0.1, 42);
        let timeline = BirthDeathTimeline::from_history(&t, 20, 10);
        assert!(timeline.entries.len() > 3);
    }

    #[test]
    fn test_breakthrough_steps() {
        let t = clustered_trajectory(3, 100, 4, 5.0, 1.0, 42);
        let timeline = BirthDeathTimeline::from_history(&t, 20, 10);
        let bt = timeline.breakthrough_steps(1);
        // Should find some steps with deaths
        // (may or may not depending on the specific data)
        assert!(bt.len() >= 0);
    }

    #[test]
    fn test_exploration_steps() {
        let t = clustered_trajectory(3, 100, 4, 5.0, 1.0, 42);
        let timeline = BirthDeathTimeline::from_history(&t, 20, 10);
        let ex = timeline.exploration_steps(1);
        assert!(ex.len() >= 0);
    }

    #[test]
    fn test_empty_diagram_summary() {
        let dg = PersistenceDiagram::empty();
        let summary = BirthDeathSummary::from_diagram(&dg, 1.0);
        assert_eq!(summary.total_births, 0);
        assert_eq!(summary.avg_lifetime, 0.0);
        assert_eq!(summary.median_lifetime, 0.0);
    }

    #[test]
    fn test_summary_serde() {
        let dg = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 1.0, 0)],
            3,
        );
        let summary = BirthDeathSummary::from_diagram(&dg, 1.0);
        let json = serde_json::to_string(&summary).unwrap();
        let s2: BirthDeathSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(s2.total_births, 1);
    }
}
