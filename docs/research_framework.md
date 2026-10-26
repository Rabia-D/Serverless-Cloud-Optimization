
# CP493: Intelligent Serverless Resource Management

## Proposed Research Framework and Methodology

**Research Title:** Comparative Evaluation of Machine Learning, LLM-Assisted Reasoning, and Reinforcement Learning for Serverless Resource Optimization

**Status:** Preliminary research framework and exploratory implementation

---

## 1. Research Objective

This research investigates intelligent resource-management strategies for serverless cloud computing environments.

The primary objective is to compare three approaches for deciding when serverless application environments should remain warm in memory or be offloaded:

1. Traditional machine learning (ML), neural network (NN), and deep learning (DL) prediction-based resource management.
2. ML-based prediction enhanced with large language model (LLM) reasoning.
3. Reinforcement learning (RL) for adaptive resource management.

The goal is to reduce cold starts and unnecessary memory consumption while maintaining system performance and minimizing decision overhead.

## 2. Research Questions

**RQ1:** How does LLM-assisted reasoning affect resource-management decisions compared with traditional ML prediction-based policies?

**RQ2:** How does RL-based adaptive resource management compare with ML-only and ML + LLM approaches?

**RQ3:** What trade-offs exist between cold-start frequency, memory utilization, decision overhead, and overall performance?

## 3. Dataset Selection

### Primary Dataset: Microsoft Azure Functions Invocation Trace 2021

The 2021 Azure Functions trace is the primary dataset because it provides more recent serverless workload information.

Available fields include:

- `app`: Anonymized application identifier
- `func`: Anonymized function identifier
- `end_timestamp`: Invocation completion time
- `duration`: Invocation execution duration

### Initial Dataset Preparation

An exploratory sample of the 10 highest-activity applications has been prepared.

Current results:

- Applications: 10
- Total invocations: 1,661,525
- Observation period: 14 days
- Aggregation interval: 5 minutes
- Total application-window records: 40,320
- Active application-windows: 12,488
- Zero-invocation windows: 27,832

The processed invocation totals were validated against the original trace.

The sample currently emphasizes high-activity applications and is not intended to represent the full diversity of Azure workloads.

### Secondary Dataset: Azure Functions 2019

The 2019 dataset may be investigated for additional resource-related features and cross-dataset generalization experiments.

Its use remains subject to reviewing feature compatibility and the supervisor's guidance.

## 4. Data Preprocessing and Feature Engineering

### Tools: Python, pandas, and Dask

Dask is incorporated to support scalable processing of the Azure invocation trace.

Proposed preprocessing workflow:

1. Load invocation records from the raw trace.
2. Group records by application and five-minute time window.
3. Count invocations within each window.
4. Insert zero-invocation windows.
5. Calculate workload characteristics.
6. Prepare chronological training, validation, and test sets.

Candidate features include:

- Invocation count in previous time windows
- Rolling average invocation rate
- Time since the last invocation
- Idle-period duration
- Workload burstiness
- Historical activity trends
- Current warm/offloaded status
- Available memory in the simulated environment

### Dask Validation

Dask was used to independently aggregate invocation records for the selected applications.

Results:

- Dask invocation total: 1,661,525
- pandas invocation total: 1,661,525
- Matching totals: Yes
- Active application-windows: 12,488

This verifies agreement between the Dask and pandas aggregation results.

Dask performance and scalability have not yet been benchmarked.

## 5. Proposed AI Framework

### Approach A: Traditional ML / NN / DL

**Purpose:** Predict future invocation activity and use those predictions to guide resource-management decisions.

Proposed inputs:

- Recent invocation history
- Rolling invocation statistics
- Idle periods
- Temporal workload features

Candidate models:

- Random Forest
- XGBoost
- LSTM
- Time-series Transformer

Proposed outputs:

- Predicted next-window invocation count
- Probability of future invocation activity

A baseline decision policy will convert predictions into keep-warm or offload priorities.

The initial implementation will focus on establishing a simple, reproducible ML baseline before investigating more complex architectures.

### Approach B: ML + LLM-Assisted Reasoning

**Purpose:** Evaluate whether LLM reasoning can improve resource-management decisions based on ML predictions.

The LLM will not directly predict invocation counts.

Instead, it will receive information such as:

- ML-predicted invocation demand
- Recent workload history
- Burstiness and idle-time characteristics
- Current warm/offloaded status
- Available memory and resource constraints

The LLM will interpret this information and propose refinements to application priorities or allocation recommendations.

Outputs will be structured so they can be validated before use.

For a fair comparison, the ML-only and ML + LLM approaches will use the same underlying predictions.

### Approach C: Reinforcement Learning

**Purpose:** Investigate whether an RL agent can learn adaptive resource-management policies from simulated feedback.

Proposed RL components:

**Environment:**
- Trace-replay simulator using Azure invocation records

**State:**
- Recent invocation counts
- Idle duration
- Current warm/offloaded status
- Available memory
- Recent resource-management outcomes

**Actions:**
- Keep an application environment warm
- Offload an application environment

**Reward:**
- Penalize simulated cold starts
- Penalize unnecessary warm-memory consumption
- Optionally penalize decision overhead

A preliminary reward formulation is:

R(t) = -[alpha * C(t) + beta * M(t) + gamma * D(t)]

Where:

- C(t): Simulated cold-start penalty
- M(t): Warm-memory consumption
- D(t): Decision overhead
- alpha, beta, gamma: Tunable objective weights

Candidate RL methods include DQN or other algorithms appropriate for the final state and action spaces.

The RL agent will be trained on earlier trace segments and evaluated on unseen workload periods.

## 6. Google OR-Tools: Resource Allocation

**Purpose:** Solve the constrained optimization problem of deciding which application environments should remain warm.

OR-Tools is not a workload prediction model.

It receives application priorities or estimated benefits and determines a feasible allocation under a limited resource budget.

Proposed optimization:

Maximize:

    sum(benefit_i * keep_warm_i)

Subject to:

    sum(memory_i * keep_warm_i) <= memory_budget

Where:

- keep_warm_i is a binary decision variable.
- benefit_i is the estimated value of keeping application i warm.
- memory_i is the application's assumed memory requirement.
- memory_budget is the total available memory.

### Initial OR-Tools Demonstration

A preliminary allocation problem was implemented using the Azure 2021 sample.

Configuration:

- 10 applications
- Maximum of 4 warm environments
- Equal-sized illustrative application slots
- Historical invocation totals as allocation scores

OR-Tools selected the four applications with the highest historical invocation totals.

This demonstrates that the optimization model can produce feasible resource allocations.

It does not yet establish cold-start reductions or predictive performance.

### Integration with the Three Approaches

**ML-only:** ML predictions determine application priorities.

**ML + LLM:** LLM reasoning refines priorities derived from the same ML predictions.

**RL:** The agent may propose priorities or actions that are constrained by the same allocation layer.

Whether RL should use OR-Tools as a feasibility layer or learn constrained actions directly remains an open design question.

## 7. Trace-Replay Simulation

A shared simulator will be developed to compare the approaches under equivalent conditions.

Proposed simulation workflow:

1. Observe workload history available at the current decision time.
2. Generate model predictions or RL decisions.
3. Determine feasible warm/offload allocations.
4. Replay subsequent invocation events.
5. Estimate cold starts and memory consumption.
6. Record decision overhead and simulated performance.
7. Update the environment state.
8. Continue to the next decision interval.

The simulator should preserve chronological ordering and prevent future workload information from entering current decisions.

The 2021 trace does not directly identify cold starts or application memory allocations.

Therefore, initial experiments will require documented assumptions for:

- Memory requirements
- Cold-start penalties
- Warm-environment retention
- Resource budgets

Original invocation timestamps may be needed for event-level cold-start evaluation, rather than relying only on five-minute aggregate counts.

## 8. Evaluation Methodology

The three approaches will be evaluated under comparable workloads, memory constraints, and simulation assumptions.

### Prediction Metrics

For forecasting models:

- MAE or RMSE for invocation-count prediction
- Precision, recall, and F1-score if predicting whether an application will be invoked

### Resource-Management Metrics

For all three approaches:

- Simulated cold-start frequency
- Warm-memory utilization
- Resource consumption
- Decision latency and computational overhead
- Overall simulated system performance

### Experimental Design

- Use chronological train/validation/test splits.
- Prevent future-data leakage.
- Apply equivalent memory budgets.
- Compare approaches on the same unseen workload periods.
- Include multiple workload patterns.
- Investigate changing traffic patterns and generalization.

The main analysis will focus on performance versus resource-cost trade-offs.

## 9. Proposed Software Stack

| Component | Tools |
|---|---|
| Data processing | Python, pandas, Dask |
| Data visualization | Matplotlib |
| ML baselines | scikit-learn, XGBoost, PyCaret |
| NN / DL | PyTorch, LSTM, time-series Transformers |
| LLM reasoning | Pretrained LLM with structured prompts |
| Reinforcement learning | Gymnasium, Stable-Baselines3 |
| Resource optimization | Google OR-Tools |
| Simulation | Custom Python trace-replay environment |
| Version control | Git and GitHub |

Some tools are currently installed or demonstrated, while others remain proposed.

## 10. Current Research Progress

Completed:

- Downloaded the Azure Functions 2021 invocation trace.
- Created a 10-application exploratory dataset.
- Aggregated invocations into five-minute windows.
- Included zero-invocation intervals.
- Validated invocation totals against the raw trace.
- Independently verified aggregation using Dask.
- Generated exploratory workload visualizations.
- Implemented a preliminary OR-Tools allocation example.
- Created a GitHub research repository.
- Defined an initial three-approach methodology.

Not yet completed:

- ML model training
- LLM-assisted decision implementation
- RL agent training
- Full trace-replay simulator
- Comparative experimental evaluation

## 11. Next Research Steps

### Phase 1: Dataset Expansion

- Include more diverse application workloads.
- Extract historical and temporal features.
- Review the Azure 2019 dataset.
- Establish chronological dataset splits.

### Phase 2: Simulation Environment

- Define memory and cold-start assumptions.
- Implement warm/offload state transitions.
- Integrate OR-Tools allocation constraints.
- Validate simulator behaviour.

### Phase 3: ML Baseline

- Train a simple forecasting model.
- Evaluate prediction accuracy.
- Connect predictions to resource decisions.

### Phase 4: LLM-Assisted Reasoning

- Design structured LLM inputs and outputs.
- Integrate LLM-generated priority refinements.
- Measure resource-management outcomes and decision overhead.

### Phase 5: Reinforcement Learning

- Define the final state and action spaces.
- Implement the reward function.
- Train an RL agent using the simulator.
- Evaluate on unseen workload periods.

### Phase 6: Comparative Evaluation

- Compare ML-only, ML + LLM, and RL approaches.
- Analyze cold-start and memory trade-offs.
- Investigate robustness to workload changes.
- Assess whether added reasoning or learning complexity improves performance.

## 12. Questions for Professor Daraghmeh

1. Should RL be implemented as a standalone resource-management approach, or should it use OR-Tools to enforce allocation constraints?

2. Since the 2021 dataset does not contain direct cold-start labels or memory allocations, should these initially be simulated, or should the 2019 dataset be used to inform the assumptions?

3. Is XGBoost an appropriate first ML baseline before introducing LSTM or time-series Transformer models?

4. Should the initial evaluation focus on application-level resource allocation with five-minute decision intervals?

5. Would a shared trace-replay simulator be an appropriate environment for comparing all three approaches?

---

**Research Status:** Preliminary methodology and proof-of-concept components. The full comparative framework remains to be implemented and evaluated.

