//! Persistence landscape — functional summary of persistence diagrams.

use crate::persistence::PersistenceDiagram;
use serde::{Deserialize, Serialize};

/// A single lambda function in the persistence landscape.
/// lambda_k(t) = k-th largest value among tent functions at point t.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandscapeFunction {
    /// (t, value) pairs defining the piecewise-linear function.
    pub points: Vec<(f64, f64)>,
    pub order: usize, // lambda_k (0-indexed)
}

impl LandscapeFunction {
    /// Evaluate the landscape function at a given t.
    pub fn evaluate(&self, t: f64) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        // Find the interval containing t
        for w in self.points.windows(2) {
            let (t0, v0) = w[0];
            let (t1, v1) = w[1];
            if t >= t0 && t <= t1 {
                if (t1 - t0).abs() < 1e-15 {
                    return v0.max(v1);
                }
                let frac = (t - t0) / (t1 - t0);
                return v0 + frac * (v1 - v0);
            }
        }
        0.0
    }

    /// L^p norm of the landscape function.
    pub fn l_p_norm(&self, p: f64) -> f64 {
        if p.is_infinite() {
            return self
                .points
                .iter()
                .map(|(_, v)| v.abs())
                .fold(0.0_f64, f64::max);
        }
        let mut sum = 0.0;
        for w in self.points.windows(2) {
            let (t0, v0) = w[0];
            let (t1, v1) = w[1];
            let dt = (t1 - t0).abs();
            // Integrate |f(t)|^p using Simpson's rule
            let vm = (v0 + v1) / 2.0;
            sum += dt / 6.0 * (v0.abs().powf(p) + 4.0 * vm.abs().powf(p) + v1.abs().powf(p));
        }
        sum.powf(1.0 / p)
    }
}

/// Persistence landscape: a sequence of landscape functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceLandscape {
    pub functions: Vec<LandscapeFunction>,
    pub dim: usize,
}

impl PersistenceLandscape {
    pub fn new(functions: Vec<LandscapeFunction>, dim: usize) -> Self {
        Self { functions, dim }
    }

    /// Compute from a persistence diagram.
    pub fn from_diagram(diagram: &PersistenceDiagram, dim: usize, k_max: usize, resolution: usize) -> Self {
        let pairs: Vec<_> = diagram
            .pairs
            .iter()
            .filter(|p| p.dim == dim && p.is_finite())
            .collect();

        if pairs.is_empty() {
            return PersistenceLandscape::new(vec![], dim);
        }

        let birth_min = pairs.iter().map(|p| p.birth).fold(f64::INFINITY, f64::min);
        let death_max = pairs.iter().map(|p| p.death).fold(f64::NEG_INFINITY, f64::max);

        let range = (death_max - birth_min).max(1e-10);
        let t_min = birth_min - 0.1 * range;
        let t_max = death_max + 0.1 * range;
        let dt = (t_max - t_min) / resolution as f64;

        let mut functions = Vec::new();

        for k in 0..k_max {
            let mut points = Vec::new();
            for i in 0..=resolution {
                let t = t_min + i as f64 * dt;
                // Evaluate k-th landscape function at t
                let mut tent_values: Vec<f64> = pairs
                    .iter()
                    .map(|p| {
                        let mid = (p.birth + p.death) / 2.0;
                        let half_h = (p.death - p.birth) / 2.0;
                        if half_h <= 0.0 {
                            0.0
                        } else {
                            (half_h - (t - mid).abs()).max(0.0)
                        }
                    })
                    .collect();
                tent_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let val = if k < tent_values.len() { tent_values[k] } else { 0.0 };
                points.push((t, val));
            }
            functions.push(LandscapeFunction {
                points,
                order: k,
            });
        }

        PersistenceLandscape::new(functions, dim)
    }

    /// Evaluate k-th landscape function at t.
    pub fn evaluate(&self, k: usize, t: f64) -> f64 {
        self.functions
            .get(k)
            .map(|f| f.evaluate(t))
            .unwrap_or(0.0)
    }

    /// L^p norm distance between two landscapes.
    pub fn distance(other: &PersistenceLandscape, us: &PersistenceLandscape, p: f64) -> f64 {
        let max_len = other.functions.len().max(us.functions.len());
        let mut total = 0.0;
        for k in 0..max_len {
            let n1 = other.functions.get(k).map(|f| f.l_p_norm(p)).unwrap_or(0.0);
            let n2 = us.functions.get(k).map(|f| f.l_p_norm(p)).unwrap_or(0.0);
            total += (n1 - n2).powf(p);
        }
        total.powf(1.0 / p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::*;

    fn simple_diagram() -> PersistenceDiagram {
        PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 2.0, 0),
                PersistencePair::new(1.0, 3.0, 0),
            ],
            4,
        )
    }

    #[test]
    fn test_landscape_from_diagram() {
        let dg = simple_diagram();
        let landscape = PersistenceLandscape::from_diagram(&dg, 0, 3, 100);
        assert_eq!(landscape.functions.len(), 3);
    }

    #[test]
    fn test_landscape_from_empty() {
        let dg = PersistenceDiagram::empty();
        let landscape = PersistenceLandscape::from_diagram(&dg, 0, 2, 50);
        assert_eq!(landscape.functions.len(), 0);
    }

    #[test]
    fn test_landscape_evaluate_peak() {
        let dg = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 2.0, 0)],
            2,
        );
        let landscape = PersistenceLandscape::from_diagram(&dg, 0, 1, 200);
        // Peak should be at midpoint (1.0) with value 1.0
        let val = landscape.evaluate(0, 1.0);
        assert!((val - 1.0).abs() < 0.1, "Expected ~1.0, got {}", val);
    }

    #[test]
    fn test_landscape_evaluate_outside() {
        let dg = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 2.0, 0)],
            2,
        );
        let landscape = PersistenceLandscape::from_diagram(&dg, 0, 1, 100);
        // Outside the support should be 0
        let val = landscape.evaluate(0, 100.0);
        assert!(val < 0.01);
    }

    #[test]
    fn test_landscape_zero_outside() {
        let dg = simple_diagram();
        let landscape = PersistenceLandscape::from_diagram(&dg, 0, 1, 100);
        // Well outside the range
        assert!(landscape.evaluate(0, -100.0) < 0.01);
    }

    #[test]
    fn test_landscape_second_function_lower() {
        let dg = PersistenceDiagram::new(
            vec![
                PersistencePair::new(0.0, 4.0, 0),
                PersistencePair::new(0.5, 3.5, 0),
            ],
            4,
        );
        let landscape = PersistenceLandscape::from_diagram(&dg, 0, 2, 200);
        // Lambda_2 should be <= Lambda_1 everywhere
        for t in (0..200).map(|i| -1.0 + i as f64 * 6.0 / 200.0) {
            let v1 = landscape.evaluate(0, t);
            let v2 = landscape.evaluate(1, t);
            assert!(v2 <= v1 + 1e-6, "Lambda_2({}) = {} > Lambda_1 = {}", t, v2, v1);
        }
    }

    #[test]
    fn test_landscape_function_evaluate() {
        let f = LandscapeFunction {
            points: vec![(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)],
            order: 0,
        };
        assert!((f.evaluate(0.5) - 1.0).abs() < 1e-10);
        assert!((f.evaluate(1.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_landscape_function_l2_norm() {
        let f = LandscapeFunction {
            points: vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
            order: 0,
        };
        // Area of triangle with height 1, width 2 → L2 norm = sqrt(2/3)
        let norm = f.l_p_norm(2.0);
        assert!(norm > 0.0);
    }

    #[test]
    fn test_landscape_function_linfinity_norm() {
        let f = LandscapeFunction {
            points: vec![(0.0, 0.0), (1.0, 5.0), (2.0, 0.0)],
            order: 0,
        };
        let norm = f.l_p_norm(f64::INFINITY);
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_landscape_function_empty() {
        let f = LandscapeFunction {
            points: vec![],
            order: 0,
        };
        assert!((f.evaluate(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_landscape_serde() {
        let dg = simple_diagram();
        let landscape = PersistenceLandscape::from_diagram(&dg, 0, 2, 50);
        let json = serde_json::to_string(&landscape).unwrap();
        let l2: PersistenceLandscape = serde_json::from_str(&json).unwrap();
        assert_eq!(l2.functions.len(), 2);
    }

    #[test]
    fn test_landscape_distance_same() {
        let dg = simple_diagram();
        let l1 = PersistenceLandscape::from_diagram(&dg, 0, 2, 50);
        let l2 = PersistenceLandscape::from_diagram(&dg, 0, 2, 50);
        let d = PersistenceLandscape::distance(&l1, &l2, 2.0);
        assert!(d < 0.01, "Distance to self should be ~0, got {}", d);
    }

    #[test]
    fn test_landscape_distance_different() {
        let dg1 = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 2.0, 0)],
            3,
        );
        let dg2 = PersistenceDiagram::new(
            vec![PersistencePair::new(0.0, 10.0, 0)],
            3,
        );
        let l1 = PersistenceLandscape::from_diagram(&dg1, 0, 1, 100);
        let l2 = PersistenceLandscape::from_diagram(&dg2, 0, 1, 100);
        let d = PersistenceLandscape::distance(&l1, &l2, 2.0);
        assert!(d > 0.1, "Different diagrams should have positive distance, got {}", d);
    }
}
