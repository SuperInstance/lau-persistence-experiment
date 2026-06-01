# lau-persistence-experiment

**Persistence diagrams of agent belief manifolds predict learning trajectories** — a topological data analysis framework for understanding how agents learn, in pure Rust.

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-149-green.svg)](#testing)

---

## What This Does

This crate tests a specific hypothesis from the PLATO/LAU research program:

> **Agents whose belief manifolds have long-lived topological features (high persistence) learn more slowly but more robustly. Agents with short-lived features learn fast but are fragile.**

It does this by applying **persistent homology** — the core tool of topological data analysis (TDA) — to the trajectory of an agent's belief states as it learns. The result is a *persistence diagram* that captures the birth and death of topological features (connected components, loops, voids) across scales.

From that diagram, the crate derives:
- **Betti curves** — how topology evolves during learning
- **Persistence landscapes** — functional summaries suitable for ML pipelines
- **Topological complexity** — a learning-difficulty predictor
- **Birth-death events** — when features appear and disappear
- **Phase transitions** — sudden topology changes = breakthroughs or failures
- **Robustness scores** — high persistence ⟹ robust, low ⟹ fragile
- **Cross-agent comparison** — do different learners have different topological signatures?
- **Falsification** — systematic search for counterexamples

Part of the **PLATO/LAU ecosystem** — a mathematically rigorous framework for building educational agents.

## Key Idea

An agent's belief state traces out a **manifold** in some high-dimensional belief space. As the agent learns, this manifold changes shape — it might split into disconnected clusters (undecided beliefs), develop loops (circular reasoning), or converge to a point (certainty).

**Persistent homology** captures this shape evolution. The key insight is that the *persistence* of a topological feature — how long it survives across scales — is a proxy for its significance:
- **Long-lived features** = stable, robust beliefs that resist perturbation
- **Short-lived features** = noise, transient beliefs, or artifacts

By computing persistence diagrams over sliding windows of a learning trajectory, you can watch topology evolve in real time. Phase transitions — sudden jumps in Betti numbers or persistence — correspond to breakthroughs (sudden understanding) or failures (collapse of belief structure).

## Install

```toml
# Cargo.toml
[dependencies]
lau-persistence-experiment = "0.1"
```

Dependencies: `nalgebra` (linear algebra), `serde`/`serde_json` (serialization).

## Quick Start

```rust
use lau_persistence_experiment::*;

// --- Generate a synthetic belief trajectory ---
let history = belief::synthetic_trajectory(3, 100, 0.1, 0.01, 42);
println!("Trajectory: {} points in {}D", history.len(), history.dimension().unwrap());

// --- Compute persistence diagram (H0 + H1) ---
let diagram = persistence::compute_persistence(&history, 1);
println!("Features found: {}", diagram.pairs.len());
println!("Max persistence: {:.3}", diagram.max_persistence());
println!("Persistence entropy: {:.3}", diagram.persistence_entropy());

// --- Betti curve (topology over time) ---
let betti_curve = betti::compute_betti_curve(&history, 20, 5, 1, 10);
println!("Max Betti-0 (disconnectedness): {}", betti_curve.max_betti0());
println!("Max Betti-1 (loops): {}", betti_curve.max_betti1());

// --- Persistence landscape (functional summary) ---
let landscape = landscape::PersistenceLandscape::from_diagram(&diagram, 0, 3, 100);
println!("Landscape norms: {:?}", landscape.l1_norms());

// --- Topological complexity ---
let complexity = complexity::TopologicalComplexity::from_history(&history, 1);
println!("Total persistence: {:.3}", complexity.total_persistence);
println!("Complexity score: {:.3}", complexity.complexity_score());

// --- Robustness assessment ---
let robustness = robustness::RobustnessScore::from_history(&history, 20, 0.5);
println!("Robustness: {:.2}/1.0", robustness.robustness);
println!("Fragility: {:?}", robustness.fragility_indicators);

// --- Spiral trajectory (converging beliefs) ---
let spiral = belief::spiral_trajectory(2, 200, 5.0, 0.01, 42);
let spiral_diag = persistence::compute_persistence(&spiral, 1);

// --- Clustered trajectory (disjoint belief modes) ---
let clustered = belief::clustered_trajectory(3, 90, 3, 5.0, 0.5, 42);
let cluster_diag = persistence::compute_persistence(&clustered, 1);

// --- Compare two agents ---
let sig1 = comparison::AgentSignature::from_history("agent-a", &history, 1);
let sig2 = comparison::AgentSignature::from_history("agent-b", &clustered, 1);
let comparison = comparison::compare_agents(&sig1, &sig2);
println!("Similarity: {:.3}", comparison.similarity);

// --- Falsification: search for counterexamples ---
let result = falsification::FalsificationResult::run_standard();
println!("Counterexamples found: {}", result.counterexamples.len());
println!("Hypothesis supported: {}", result.hypothesis_supported);
```

## API Reference

### Belief Manifold Construction (`belief`)

```rust
/// A single belief state — a point in belief space.
pub struct BeliefState {
    pub coords: Vec<f64>,
    pub step: usize,
}

impl BeliefState {
    pub fn new(coords: Vec<f64>, step: usize) -> Self;
    pub fn dimension(&self) -> usize;
    pub fn distance_to(&self, other: &BeliefState) -> f64;
    pub fn to_vector(&self) -> DVector<f64>;
}

/// History of agent belief states over learning steps.
pub struct StateHistory {
    pub states: Vec<BeliefState>,
}

impl StateHistory {
    pub fn new(states: Vec<BeliefState>) -> Self;
    pub fn len(&self) -> usize;
    pub fn dimension(&self) -> Option<usize>;
    pub fn slice_by_step(&self, lo: usize, hi: usize) -> StateHistory;
    pub fn distance_matrix(&self) -> DMatrix<f64>;
    pub fn centroid(&self) -> Option<BeliefState>;
    pub fn spread(&self) -> f64;
}

// Trajectory generators
pub fn synthetic_trajectory(dim: usize, steps: usize, drift: f64, noise: f64, seed: u64) -> StateHistory;
pub fn spiral_trajectory(dim: usize, steps: usize, radius: f64, noise: f64, seed: u64) -> StateHistory;
pub fn clustered_trajectory(dim: usize, steps: usize, n_clusters: usize, cluster_spread: f64, noise: f64, seed: u64) -> StateHistory;
```

### Persistence Diagrams (`persistence`)

```rust
/// A single (birth, death, dim) point in a persistence diagram.
pub struct PersistencePair {
    pub birth: f64,
    pub death: f64,
    pub dim: usize,
}

impl PersistencePair {
    pub fn persistence(&self) -> f64;  // death - birth
    pub fn midpoint(&self) -> f64;
    pub fn is_finite(&self) -> bool;
}

/// A full persistence diagram.
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
    pub n_points: usize,
}

impl PersistenceDiagram {
    pub fn empty() -> Self;
    pub fn dimension(&self, dim: usize) -> Vec<&PersistencePair>;
    pub fn max_persistence(&self) -> f64;
    pub fn total_persistence(&self, power: f64) -> f64;
    pub fn significant_features(&self, threshold: f64) -> usize;
    pub fn avg_persistence(&self) -> f64;
    pub fn persistence_entropy(&self) -> f64;
}

// Core computation
pub fn compute_persistence(history: &StateHistory, max_dim: usize) -> PersistenceDiagram;
pub fn sliding_window_persistence(history: &StateHistory, window_size: usize, stride: usize, max_dim: usize) -> Vec<(usize, PersistenceDiagram)>;

// Distances between diagrams
pub fn bottleneck_distance(dg1: &PersistenceDiagram, dg2: &PersistenceDiagram) -> f64;
pub fn wasserstein_distance(dg1: &PersistenceDiagram, dg2: &PersistenceDiagram, p: f64) -> f64;
```

### Betti Curves (`betti`)

```rust
pub struct BettiSnapshot {
    pub step: usize,
    pub filtration: f64,
    pub betti: Vec<usize>,
}

pub struct BettiCurve {
    pub snapshots: Vec<BettiSnapshot>,
    pub max_dim: usize,
}

impl BettiCurve {
    pub fn betti_at_step(&self, step: usize, k: usize) -> Option<usize>;
    pub fn max_betti0(&self) -> usize;
    pub fn max_betti1(&self) -> usize;
    pub fn betti0_area(&self) -> f64;
    pub fn betti1_area(&self) -> f64;
}

pub fn betti_from_diagram(diagram: &PersistenceDiagram, filtration: f64, max_dim: usize) -> Vec<usize>;
pub fn compute_betti_curve(history: &StateHistory, window_size: usize, stride: usize, max_dim: usize, n_filtration_steps: usize) -> BettiCurve;
pub fn betti_curve_filtration(diagram: &PersistenceDiagram, max_dim: usize, n_steps: usize) -> BettiCurve;
pub fn euler_characteristic_curve(curve: &BettiCurve) -> Vec<(usize, i64)>;
```

### Persistence Landscapes (`landscape`)

```rust
pub struct LandscapeFunction {
    pub points: Vec<(f64, f64)>,
    pub order: usize,  // lambda_k (0-indexed)
}

impl LandscapeFunction {
    pub fn evaluate(&self, t: f64) -> f64;
}

pub struct PersistenceLandscape {
    pub functions: Vec<LandscapeFunction>,
}

impl PersistenceLandscape {
    pub fn from_diagram(diagram: &PersistenceDiagram, dim: usize, max_order: usize, resolution: usize) -> Self;
    pub fn evaluate(&self, t: f64) -> Vec<f64>;
    pub fn l1_norms(&self) -> Vec<f64>;
    pub fn l2_norms(&self) -> Vec<f64>;
    pub fn l_inf_norms(&self) -> Vec<f64>;
}
```

### Topological Complexity (`complexity`)

```rust
pub struct TopologicalComplexity {
    pub total_persistence: f64,
    pub total_persistence_sq: f64,
    pub h0_significant: usize,
    pub h1_significant: usize,
    pub max_persistence: f64,
    pub persistence_entropy: f64,
    pub complexity_score: f64,
}

impl TopologicalComplexity {
    pub fn from_history(history: &StateHistory, max_dim: usize) -> Self;
}

pub struct ComplexityTrajectory {
    pub trajectory: Vec<(usize, TopologicalComplexity)>,
}

impl ComplexityTrajectory {
    pub fn from_history(history: &StateHistory, window_size: usize, stride: usize, max_dim: usize) -> Self;
    pub fn max_complexity(&self) -> f64;
    pub fn complexity_change_rate(&self) -> Vec<(usize, f64)>;
}
```

### Birth-Death Analysis (`birth_death`)

```rust
pub enum BirthDeathType { Birth, Death, Merge, Split }

pub struct BirthDeathEvent {
    pub step: usize,
    pub birth_time: f64,
    pub death_time: f64,
    pub dim: usize,
    pub persistence: f64,
    pub event_type: BirthDeathType,
}

pub struct BirthDeathTimeline {
    pub events: Vec<BirthDeathEvent>,
}

impl BirthDeathTimeline {
    pub fn from_history(history: &StateHistory, window_size: usize, stride: usize, max_dim: usize) -> Self;
    pub fn births(&self) -> Vec<&BirthDeathEvent>;
    pub fn deaths(&self) -> Vec<&BirthDeathEvent>;
    pub fn most_persistent(&self, n: usize) -> Vec<&BirthDeathEvent>;
}
```

### Phase Transitions (`phase`)

```rust
pub enum TransitionType { Breakthrough, Collapse, Bifurcation, Convergence }

pub struct PhaseTransition {
    pub step: usize,
    pub transition_type: TransitionType,
    pub magnitude: f64,
    pub description: String,
}

pub struct PhaseTransitionDetector {
    pub transitions: Vec<PhaseTransition>,
}

impl PhaseTransitionDetector {
    pub fn detect(history: &StateHistory, window_size: usize, stride: usize, threshold: f64) -> Self;
    pub fn breakthroughs(&self) -> Vec<&PhaseTransition>;
    pub fn collapses(&self) -> Vec<&PhaseTransition>;
}
```

### Robustness (`robustness`)

```rust
pub struct RobustnessScore {
    pub robustness: f64,           // 0 = fragile, 1 = robust
    pub mean_persistence: f64,
    pub long_lived_fraction: f64,
    pub stability: f64,
    pub fragility_indicators: Vec<String>,
}

impl RobustnessScore {
    pub fn from_history(history: &StateHistory, window_size: usize, threshold: f64) -> Self;
}
```

### Cross-Agent Comparison (`comparison`)

```rust
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
    pub fn from_history(name: &str, history: &StateHistory, max_dim: usize) -> Self;
}

pub struct ComparisonResult {
    pub agent_a: String,
    pub agent_b: String,
    pub similarity: f64,
    pub bottleneck_dist: f64,
    pub wasserstein_dist: f64,
}

pub fn compare_agents(sig1: &AgentSignature, sig2: &AgentSignature) -> ComparisonResult;
pub fn rank_agents(agents: &[AgentSignature]) -> Vec<(String, f64)>;
```

### Falsification (`falsification`)

```rust
pub struct CounterexampleCandidate {
    pub name: String,
    pub persistence_score: f64,
    pub robustness_score: f64,
    pub learning_speed: f64,
    pub is_counterexample: bool,
    pub reason: String,
}

pub struct FalsificationResult {
    pub candidates: Vec<CounterexampleCandidate>,
    pub counterexamples: Vec<CounterexampleCandidate>,
    pub hypothesis_supported: bool,
}

impl FalsificationResult {
    pub fn run_standard() -> Self;
}
```

## How It Works

### 1. Belief Manifold Construction

Agent states (belief vectors) are collected over learning steps into a `StateHistory`. Three trajectory generators produce canonical learning patterns:
- **Synthetic**: random walk with drift and noise (generic learning)
- **Spiral**: converging to a point (successful convergence)
- **Clustered**: jumping between modes (exploration/exploitation)

Pairwise Euclidean distances form a distance matrix, which is the input to persistent homology.

### 2. Vietoris-Rips Persistent Homology

The core algorithm builds a **Vietoris-Rips complex** from the distance matrix:
- **H0** (connected components): computed via union-find over sorted edges (single-linkage clustering). Each merge produces a persistence pair `(0, d_merge)`. One essential component persists to infinity.
- **H1** (loops): for each triple of points, the cycle is born when the third edge enters the complex and dies when the triangle fills. Simplified model takes the most persistent loops up to `n/3`.

### 3. Sliding Window Analysis

Persistence diagrams are computed over sliding windows of the trajectory, producing a time series of topological summaries. This reveals *when* features appear and disappear during learning.

### 4. Derivative Analyses

From the persistence diagrams, the crate derives:
- **Betti curves**: count alive features at each filtration value, tracked over windows
- **Landscapes**: convert diagrams to piecewise-linear functions λ_k(t) — the k-th largest "tent" at each point
- **Complexity**: aggregate metrics (total persistence, entropy, significant feature counts) that predict learning difficulty
- **Phase transitions**: detect sudden jumps in Betti numbers or complexity (threshold-based)
- **Robustness**: composite score from persistence, stability across windows, and fragility indicators
- **Falsification**: generate diverse synthetic agents and check if any violates the hypothesis

## The Math

### Persistent Homology

Given a point cloud X = {x₁, ..., xₙ} with distance function d, the **Vietoris-Rips complex** at scale ε has a k-simplex [x_{i₀}, ..., x_{i_k}] whenever all pairwise distances are ≤ ε.

As ε increases from 0 to ∞, the complex grows. **Persistent homology** tracks:
- **Birth**: when a homology class (component, loop, void) first appears
- **Death**: when it merges with an older class or is filled in

The result is a **persistence diagram** — a multiset of points (bᵢ, dᵢ) in the plane. Points far from the diagonal (high persistence) represent significant features; points near the diagonal are noise.

### Betti Numbers

The **k-th Betti number** βₖ at filtration value ε counts the number of k-dimensional holes:
- β₀ = number of connected components
- β₁ = number of loops
- β₂ = number of voids

From the persistence diagram:

```
βₖ(ε) = |{(bᵢ, dᵢ) : bᵢ ≤ ε < dᵢ, dim = k}|
```

### Persistence Landscape

The landscape converts a diagram into a sequence of functions. For each pair (bᵢ, dᵢ), define the tent function:

```
Λᵢ(t) = max(0, min(t - bᵢ, dᵢ - t))
```

Then λₖ(t) = k-th largest value among {Λᵢ(t)}. Landscapes live in a Banach space — you can take averages, compute distances, and use them as features in ML.

### Distance Metrics

**Bottleneck distance** (stable under perturbation):

```
d_B(D₁, D₂) = inf { ε : ∃ matching M, ∀(p,q) ∈ M, ||p-q||∞ ≤ ε }
```

Simplified implementation matches each pair to its nearest neighbor (same dimension) or the diagonal.

**Wasserstein distance** (finer-grained):

```
W_p(D₁, D₂) = (inf_M Σ ||p-q||∞ᵖ)^{1/p}
```

### Persistence Entropy

Treating persistences as a probability distribution:

```
H = -Σ (pᵢ/p_total) · ln(pᵢ/p_total)
```

High entropy = many features with similar persistence. Low entropy = dominated by one or few features.

### Euler Characteristic Curve

```
χ(ε) = Σₖ (-1)ᵏ βₖ(ε)
```

Tracks the alternating sum of Betti numbers across filtration values.

## Testing

**149 tests** covering all modules:

| Module | Area | Focus |
|---|---|---|
| `belief` | 27 tests | State construction, distances, centroids, trajectories, serialization |
| `persistence` | 30 tests | Pair/diagram ops, H0/H1 computation, clustering, loops, sliding windows, distances |
| `betti` | 18 tests | Betti numbers, curves, areas, Euler characteristic, serialization |
| `landscape` | 15 tests | Landscape construction, evaluation, norms |
| `complexity` | 15 tests | Complexity metrics, trajectories, change rates |
| `birth_death` | 15 tests | Birth/death events, timelines, filtering |
| `phase` | 12 tests | Transition detection, breakthroughs, collapses |
| `robustness` | 10 tests | Robustness scores, fragility indicators |
| `comparison` | 7 tests | Agent signatures, ranking, similarity |
| `falsification` | 10 tests | Counterexample search, hypothesis validation |

Run with:

```bash
cargo test
```

## License

MIT
