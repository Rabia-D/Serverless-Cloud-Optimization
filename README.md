# CP493: Intelligent Serverless Resource Management

## Research Overview
This project investigates intelligent resource management for serverless cloud environments using the Microsoft Azure Functions 2021 dataset.

## Research Objective
Compare three approaches for deciding whether to keep serverless application environments warm or offload them:

1. Machine Learning (ML): Predict future invocation activity and apply a resource management policy.
2. ML + LLM: Use ML predictions with an LLM reasoning layer to refine decisions.
3. Reinforcement Learning (RL): Learn adaptive keep-warm/offload decisions through simulated feedback.

## Dataset
Microsoft Azure Functions Invocation Trace 2021.

Initial exploratory sample:
- 10 high-activity applications
- 5-minute invocation windows
- 40,320 records
- 27,832 zero-invocation windows

The sample is exploratory and not representative of all Azure workloads.

## Proposed Evaluation Metrics
- Prediction accuracy for ML models
- Simulated cold-start frequency
- Memory utilization
- Decision-making overhead
- Overall simulated performance

## Proposed Tools
- Python and pandas
- Dask
- PyCaret
- Google OR-Tools
- Stable-Baselines3

## Current Progress
- Downloaded the Azure 2021 trace
- Created a 10-application sample
- Aggregated invocations into 5-minute windows
- Added zero-invocation windows
- Designed a proposed ML / ML + LLM / RL framework

## Next Steps
1. Validate preprocessing and time windows.
2. Analyze invocation patterns.
3. Design a trace-replay simulator.
4. Implement an ML baseline.
5. Compare ML + LLM and RL approaches.

## Status
Research framework and initial dataset preparation only.
Models have not yet been trained or evaluated.
