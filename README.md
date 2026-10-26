# CP493: Intelligent Serverless Resource Management

**Undergraduate Directed Research | Faculty of Science, Wilfrid Laurier University**

**Research Supervisor:** Professor Mustafa Daraghmeh

## Research Overview

This research project, conducted under the Faculty of Science at Wilfrid Laurier University, investigates intelligent resource management for serverless cloud computing environments using the Microsoft Azure Functions 2021 dataset.

The research explores how machine learning, large language models, and reinforcement learning can improve resource allocation decisions while balancing cold-start frequency, memory utilization, and system performance.

## Research Questions

1. How does LLM-assisted reasoning improve resource-management decisions compared to traditional ML prediction-based approaches?
2. How does reinforcement learning compare with ML-only and ML + LLM approaches for adaptive serverless resource management?
3. What trade-offs exist between cold-start frequency, memory utilization, decision overhead, and overall system performance?

## Research Objective

Compare three approaches for deciding whether to keep serverless application environments warm or offload them:

1. **Machine Learning (ML / NN / DL):** Predict future invocation activity and apply a resource-management policy.
2. **ML + LLM-Assisted Reasoning:** Use ML predictions with an LLM reasoning layer to refine resource allocation decisions.
3. **Reinforcement Learning (RL):** Learn adaptive keep-warm/offload decisions through simulated feedback.

## Dataset

**Primary Dataset:** Microsoft Azure Functions Invocation Trace 2021.

Initial exploratory sample:
- 10 high-activity applications
- 1,661,525 invocations
- 5-minute invocation windows
- 40,320 application-window records
- 27,832 zero-invocation windows

The sample is exploratory and not representative of all Azure workloads. The Azure Functions 2019 dataset may also be investigated for additional features and generalization testing.

## Proposed Research Framework

1. **Data Preprocessing:** Process Azure invocation traces using Python, pandas, and Dask.
2. **Workload Prediction:** Use ML, neural networks, or deep learning to predict future invocation activity.
3. **LLM-Assisted Reasoning:** Investigate whether LLMs can refine resource-management decisions based on ML predictions.
4. **Reinforcement Learning:** Explore adaptive resource allocation through feedback-driven learning.
5. **Resource Optimization:** Use Google OR-Tools to enforce memory constraints and optimize warm/offload allocations.
6. **Evaluation:** Compare approaches using a shared trace-replay simulation environment.

## Proposed Evaluation Metrics

- Invocation prediction accuracy
- Simulated cold-start frequency
- Memory utilization
- Decision-making overhead
- Overall simulated system performance

## Tools and Technologies

- **Python, pandas, Dask:** Dataset preprocessing and feature engineering
- **Matplotlib:** Workload visualization
- **PyCaret / scikit-learn / XGBoost:** ML baselines
- **PyTorch:** Neural networks and deep learning
- **Google OR-Tools:** Constrained resource allocation
- **Stable-Baselines3:** Proposed reinforcement learning experiments
- **Git / GitHub:** Version control and research documentation

## Current Progress

- Downloaded and processed the Azure Functions 2021 invocation trace.
- Created and validated an exploratory sample of 10 applications.
- Aggregated invocations into five-minute windows, including zero-invocation intervals.
- Verified that processed invocation totals match the original dataset.
- Independently validated invocation aggregation using Dask.
- Visualized workload patterns, including sudden invocation spikes.
- Implemented a preliminary Google OR-Tools resource-allocation demonstration.
- Developed an initial ML / ML + LLM / RL research framework.

## Next Steps

1. Expand dataset sampling to include diverse workload patterns.
2. Extract features for invocation prediction and adaptive decision-making.
3. Develop a trace-replay simulator with memory and cold-start assumptions.
4. Implement and evaluate an ML prediction baseline.
5. Integrate LLM-assisted resource-management decisions.
6. Develop an RL agent and define its state, actions, and reward function.
7. Compare the three approaches under consistent resource constraints.

## Research Status

**Ongoing Undergraduate Directed Research — Wilfrid Laurier University, Faculty of Science**

Dataset preprocessing, exploratory analysis, and initial resource-optimization demonstrations have been completed. ML, LLM-assisted reasoning, and RL experiments are currently in the proposed methodology stage.
