# lau-persistence-experiment

Tests the hypothesis that the persistence diagram of an agent's belief manifold predicts its learning trajectory.

## Hypothesis

Agents whose belief manifolds have long-lived topological features (high persistence) learn more slowly but more robustly. Agents with short-lived features learn fast but are fragile.

## Modules

- **belief** — Belief manifold construction from agent state history
- **persistence** — Persistence diagram computation (Vietoris-Rips on belief samples)
- **betti** — Betti curve tracking (how topology changes as learning progresses)
- **landscape** — Persistence landscape (functional summary of persistence diagrams)
- **complexity** — Topological complexity as a learning difficulty predictor
- **birth_death** — Birth-death analysis: when features appear/disappear during learning
- **phase** — Phase transition detection: sudden topology changes = breakthroughs or failures
- **robustness** — Robustness score: high persistence = robust, low = fragile
- **comparison** — Cross-agent comparison of topological signatures
- **falsification** — Falsification framework: counterexample search

## Usage

```rust
use lau_persistence_experiment::*;

let history = belief::synthetic_trajectory(3, 100, 0.1, 0.01, 42);
let diagram = persistence::compute_persistence(&history, 1);
let landscape = landscape::PersistenceLandscape::from_diagram(&diagram, 0, 3, 100);
let robustness = robustness::RobustnessScore::from_history(&history, 20, 0.5);
let result = falsification::FalsificationResult::run_standard();
```

## Tests

149 tests covering all modules. Run with `cargo test`.
