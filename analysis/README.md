# Analysis

Evaluation harnesses, Optuna optimization, and graph diagnostics.

## Cross‑Validation

`analysis/cross_validation.py` runs repeated CV over a dataset and models.

Key properties:
- Fold‑safe scaling (fit on train only).
- `kfold` or repeated random splits (`CVConfig`).
- Supports classification and regression tasks.

The module's intended usage is through the `CrossValidator` class wrapped over a model/loader pairing.

## Hyperparameter Optimization

`analysis/hyperparameter.py` provides Optuna wrappers with:

- Standard evaluator (single model).
- Nested CV helper (run_nested_cv).
- Early stopping by stagnation.

Provides support for config caching to user-defined paths. 

## Graph Metrics

`analysis/graph_metrics.py` computes adjacency diagnostics:

- `avg_degree`: The mean node degree of the binarized graph. Higher means denser connectivity implying more smoothing power in post processing. 
- `edge_homophily`: Fraction of edges connecting same true label. Higher indicates the topology aligns with ground-truth class structure. 
- `adjusted_homophily`: Edge homophily adjusted for class imbalance. Positive implies we can expect the topology to match more same-label than in a random graph. 
- `pred_edge_homophily`: Homophily computed on predicted labels. Higher implies predictions are spatially consistent on graph. 
- `prob_edge_l2`: Mean L2 distance of probability vectors across edges (from probability matrix ouput by meta-learner). Higher suggests neighbors have a high rate of disagreeing in their predicted probabilities. 
- `confidence_edge_diff`: Mean absolute difference in confidence across edges. Lower implies confidence is locally consistent while higher means more noisy or long, abstract edges. 
- `corrective_edge_ratio`: Among edges where predictions disagree, fraction that actually share the same true label. A higher value means the graph has a higher capacity to correct errors. 
- `recoverable_error_rate`: Percentage of incorrect nodes that have at least one neighbor with the correct label. Higher implying a better capacity to rectify these mislabelings. 
- `distance_weighted_recoverable_error_rate`: Same as `recoverable_error_rate` but weighted by geographic distance. Higher implies that errors can be better "fixed" by nearby neighbors. That is, the graph is spatially realistic. 
- `smoothness_gap`: Predicted smoothness minus true-label smoothness. Positive means predictions are rougher than ground truth implying the graph can help while a negative value suggests the graph over-smooths predictions. 
- `localicty_ratio`: Fraction of edges below a distance threshold. Higher meaning a more spatially-realistic topology. 
- `train_neighbor_coverage`: Fraction of test nodes with at least one neighbor from the training split. Higher indicates the Correct-and-Smooth training will have better reach. 

For quantifying and understanding Correct‑and‑Smooth behavior.

## Feature Diagnostics 

- `analyis/boruta.py`: feature relevance, noise determination. 
- `analysis/vif.py`: multicollinearity checks. 
