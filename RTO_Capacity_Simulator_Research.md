# RTO Capacity Simulator — Research Summary

**Purpose:** This document catalogues all simulation, ML, and optimization approaches discussed for the RTO Capacity Simulator project, with honest assessments of when each adds genuine value versus when it's overkill.

---

## 1. Problem Framing

The core question: given N employees, M seats, and a set of hybrid work policies, what is the probability distribution of seat demand on any given day? And when does demand exceed capacity?

This is fundamentally a **stochastic occupancy problem**. The approaches below range from closed-form math to full agent-based simulation, each trading off simplicity against behavioral realism.

---

## 2. Simulation Approaches

### 2.1 Analytical / Closed-Form Models

#### 2.1.1 Binomial Occupancy Model

**What it is:** Treat each employee as an independent Bernoulli trial with probability `p` of attending on a given day. Total occupancy follows a Binomial(N, p) distribution.

**How it works:**
- For each employee `e` on day `d`, compute attendance probability `p_e` based on WFH policy, team mandates, day-of-week weight, and compliance rate.
- Expected occupancy = Σ(p_e) across all employees.
- Variance = Σ(p_e × (1 - p_e)) across all employees.
- Overflow probability = P(X > seats), computed via normal approximation: `1 - Φ((seats - μ) / σ)`.

**Strengths:**
- Instant computation — no sampling, no simulation loop.
- Zero noise in results — exact analytical answers.
- Excellent baseline to validate other models against.

**Limitations:**
- Assumes independence between employees (unrealistic — teams tend to cluster).
- No feedback loops — doesn't model "I saw it was crowded yesterday, so I'll stay home today."
- Static probabilities — doesn't capture behavioral drift over time.

**Verdict:** Start here. It's the baseline every other model should be compared against.

---

#### 2.1.2 Poisson Approximation

**What it is:** When N is large and p is small-to-moderate, Binomial(N, p) ≈ Poisson(λ = Np). Simplifies overflow calculations.

**When to use over Binomial:** Mostly interchangeable at the scale of this problem (N=500). Poisson is slightly simpler for combining independent groups (sum of Poissons is Poisson), which helps when aggregating team-level attendance.

**Verdict:** Minor variant of Binomial. Use whichever is computationally convenient.

---

#### 2.1.3 Queueing Theory (M/G/c/K Model)

**What it is:** Model seats as `c` servers in a queueing system. Employees "arrive" according to a stochastic process, occupy a seat for a duration, and leave. If all seats are occupied, arrivals are either rejected (loss model) or wait (queue model).

**How it works:**
- Arrival process: employees arrive throughout the morning (e.g., Poisson process with rate λ between 8-10 AM).
- Service time: duration in office (e.g., 6-9 hours, normally distributed).
- K = total seats (finite capacity).
- Steady-state metrics: rejection probability (Erlang-B), average utilization, expected queue length.

**Strengths:**
- Captures time-of-day dynamics — peak morning demand vs. afternoon availability.
- Models seat turnover (someone leaves at 2 PM, freeing a seat for a late arriver).
- Well-established mathematical framework with known solutions.

**Limitations:**
- Requires assumptions about arrival/departure distributions that may be hard to validate.
- More complex than needed if the question is simply "how many people total, not when."
- Steady-state assumptions may not hold for a single-day analysis.

**Verdict:** Valuable if the question evolves to include staggered arrivals, hot-desking with time slots, or seat booking systems. Overkill for the basic "how many seats on a given day" question.

---

### 2.2 Monte Carlo Simulation

#### 2.2.1 Basic (Independent) Monte Carlo

**What it is:** For each simulated day, draw attendance for every employee independently from Bernoulli(p_e). Repeat for thousands of runs. Build empirical distributions of daily occupancy.

**How it works:**
- For each of 5,000 simulation runs, for each working day in the target month:
  - For each employee, flip a biased coin with probability p_e (computed from policy + day-of-week + compliance).
  - Count total attendance.
  - Record whether it exceeds seat capacity.
- Aggregate across runs: mean, std dev, percentiles (5th, 25th, 50th, 75th, 95th), overflow probability.

**Strengths:**
- Flexible — easy to add new factors without rederiving formulas.
- Produces full empirical distributions, not just mean and variance.
- Simple to implement and debug.

**Limitations:**
- Same independence assumption as Binomial (without the correlation extension below).
- Requires many runs for stable tail estimates (overflow probability).
- Slower than analytical (but still sub-second for N=500, runs=5000 with vectorized NumPy).

**Verdict:** The workhorse. This is the primary simulation backend for v1.

---

#### 2.2.2 Correlated Monte Carlo (Copula-Based)

**What it is:** Extension of basic Monte Carlo that models dependencies between employees' attendance decisions using a Gaussian copula or hierarchical sampling.

**How it works (hierarchical approach):**
- For each team on each day, draw a team-level "social factor" `s_team` from Beta(α, β).
- Each employee's attendance probability for that day becomes: `p_e_adjusted = p_e × s_team`.
- This creates positive correlation within teams: when `s_team` is high, the whole team is more likely to show up.

**How it works (copula approach):**
- Define a correlation matrix across employees (block-diagonal by team).
- Draw correlated uniform random variables using a Gaussian copula.
- Transform to Bernoulli outcomes using each employee's marginal probability.

**Strengths:**
- Captures the critical real-world dynamic: teams cluster their attendance.
- Produces more realistic tail behavior — correlated arrivals create fatter tails (higher peak days) than independent models predict.
- The hierarchical approach is simple to implement and intuitive to explain.

**Limitations:**
- Correlation structure must be assumed or calibrated — hard to get right without data.
- Copula approach is more complex to implement and explain.

**Verdict:** The hierarchical team-correlation extension is highly recommended for v1 Monte Carlo. It's simple, adds realism, and the difference in overflow estimates vs. independent Monte Carlo is a great talking point.

---

### 2.3 Discrete Event Simulation (SimPy)

**What it is:** A process-based simulation where entities (employees) interact with resources (seats) over continuous time. Unlike Monte Carlo which asks "how many people total today," DES asks "what happens minute-by-minute as people arrive and leave."

**How it works:**
- Define the office as a `simpy.Resource(capacity=total_seats)`.
- Each employee is a process: arrive at a stochastic time (e.g., Normal(9:00, 0:30)), request a seat, occupy it for a stochastic duration (e.g., Normal(7hrs, 1hr)), release it.
- If no seats available at arrival: employee is either turned away (logs a rejection) or waits.
- Run the simulation for each day, collect metrics: peak concurrent occupancy, rejection count, average wait time, utilization over time.

**Library:** `SimPy` (Python, standard for DES).

**Strengths:**
- Captures temporal dynamics that snapshot models miss entirely.
- Can model hot-desking, shift patterns, meeting room bookings, and other time-dependent resources.
- Answers questions like "what time of day do we hit capacity?" and "if we stagger team arrivals, does it help?"

**Limitations:**
- Significantly more complex to implement and parameterize than Monte Carlo.
- Requires arrival/departure time distributions — data that may not be available.
- Slower to run — each simulated day is a full event-driven simulation, not a single random draw.

**Verdict:** Excellent Phase 4 addition. Adds genuine value for facilities teams thinking about time-of-day management, staggered schedules, and real-time seat booking. Not needed for the core "how many seats" question.

---

### 2.4 Agent-Based Modeling (Mesa)

#### 2.4.1 Rule-Based ABM

**What it is:** Each employee is an autonomous agent with memory and simple decision rules. Agents observe the environment (yesterday's crowding, whether their manager attended, team norms) and adapt their attendance probability over time.

**How it works:**
- Each agent has a base attendance probability and a set of behavioral rules:
  - "If yesterday's occupancy > 90%, reduce my probability by 0.1 tomorrow."
  - "If my manager agent attended today, increase my probability by 0.15 tomorrow."
  - "If fewer than 3 teammates are in, reduce my probability by 0.1."
- The simulation runs day-by-day. Each day, agents make attendance decisions, observe outcomes, and update their internal state.
- Over multiple weeks, the system settles into an emergent equilibrium that reflects the collective adaptive behavior.

**Library:** `Mesa` (Python, standard for ABM).

**Strengths:**
- Captures feedback loops and emergent behavior that no static probability model can.
- Models policy resistance: a "3-day office mandate" might equilibrate to 2.5 days because of crowding avoidance.
- Produces rich dynamics: organic load-balancing (people shift away from peak days), herd behavior, norm formation.
- Visually compelling — agent trajectories over time make for great presentations.

**Limitations:**
- Rule design is somewhat arbitrary without behavioral data to calibrate against.
- Harder to validate — "emergent behavior" can also mean "unpredictable bugs."
- Slower than Monte Carlo due to agent-level state management.

**Verdict:** Strong addition for demonstrating systems thinking and behavioral dynamics. Best positioned as a "what happens over the first 3 months after a policy change" tool rather than a steady-state estimator.

---

#### 2.4.2 Neural-Network-Brained ABM

**What it is:** Same as above, but each agent's decision function is a small neural network instead of hand-crafted rules.

**Assessment: Overkill.** The neural network adds complexity without clear benefit:
- No training data exists for individual employee decision-making processes.
- The NN is a black box — you lose the interpretability that makes ABM compelling.
- Simple probabilistic rules with memory produce the same emergent dynamics.
- An interviewer asking "why not a lookup table?" would be hard to answer convincingly.

**Verdict:** Skip this. Rule-based ABM with ML-calibrated parameters (see Section 3.2) is the better approach — it gets the "ML" story without the unjustified complexity.

---

### 2.5 System Dynamics Model

**What it is:** A differential-equation-based approach modeling feedback loops at an aggregate level (not per-employee). Think "stocks and flows": occupancy rate drives employee satisfaction, which drives attendance willingness, which drives occupancy rate.

**How it works:**
- Define state variables: average attendance rate, employee satisfaction with office experience, perceived crowding.
- Define feedback loops:
  - Positive: high attendance → more collaboration → higher satisfaction → higher attendance.
  - Negative: high attendance → crowding → lower satisfaction → lower attendance.
- Solve the ODEs to find equilibrium points and dynamic trajectories.

**Tools:** `PySD`, `STELLA`, or just `scipy.integrate.odeint`.

**Strengths:**
- Excellent for strategic "big picture" analysis — what equilibrium does the system settle at?
- Fast computation (ODE solver, not per-agent simulation).
- Good for communicating feedback loop dynamics to non-technical stakeholders.

**Limitations:**
- Aggregate level only — no per-team or per-employee granularity.
- Requires calibrating feedback strengths, which is speculative without data.
- Not suitable for answering "what happens next Tuesday specifically?"

**Verdict:** Interesting academically. Lower priority than Monte Carlo, SimPy, or ABM for this project's goals. Could be a nice addition for executive-level "what equilibrium does hybrid work settle at" presentations.

---

## 3. Machine Learning Approaches

### 3.1 Attendance Prediction (XGBoost / LightGBM)

**What it is:** Train a gradient-boosted model on historical attendance data to predict P(in-office) for each employee on each day. This replaces the hand-tuned probability assumptions used in the simulation backends.

**Features:**
- Day-of-week (one-hot encoded).
- Employee attributes: team, seniority, role, distance from office.
- Temporal features: was this employee in-office yesterday? How many days this week so far?
- Environmental: weather (temperature, rain), holiday proximity.
- Social: is the employee's manager in-office? How many teammates are in?

**Target:** Binary — did the employee badge in (1) or not (0)?

**Output:** Calibrated probability P(in-office), which feeds directly into Monte Carlo or ABM as the attendance probability.

**Strengths:**
- Replaces assumptions with data-driven probabilities — the single biggest accuracy improvement.
- XGBoost/LightGBM handle mixed feature types, missing data, and non-linear interactions well.
- Feature importance reveals what actually drives attendance (weather? manager presence? day-of-week?).

**Limitations:**
- Requires historical badge/access data — not always available or clean.
- Model trained on past behavior may not generalize to new policies (distribution shift).
- Cold-start problem for new employees.

**Data requirements:** At minimum, 3-6 months of daily badge-in data per employee.

**Verdict:** This is the highest-value ML addition. If real attendance data is available, this should be the priority ML investment. Even with synthetic data for a portfolio project, demonstrating the pipeline (feature engineering → model training → calibrated probability → simulation input) is very compelling.

---

### 3.2 Behavioral Clustering (GMM / K-Means)

**What it is:** Cluster employees into behavioral archetypes based on attendance patterns, then assign different simulation parameters to each cluster.

**How it works:**
- Feature vector per employee: attendance rate by day-of-week (5 values), week-to-week consistency, response to crowding, streak length (consecutive days in/out).
- Apply Gaussian Mixture Model (GMM) or K-Means to discover natural clusters.
- Expected archetypes:
  - "Always in" — high attendance every day (10-15% of employees).
  - "Strict policy follower" — exactly N days per week, consistent pattern.
  - "Friday avoider" — office Mon-Thu, rarely Friday.
  - "Team-driven" — attendance correlates with teammates, not fixed schedule.
  - "Minimal presence" — comes in only for mandatory days.

**Strengths:**
- Makes simulation more realistic without per-employee modeling (cluster-level parameters).
- Produces actionable HR insights — "40% of employees are Friday avoiders."
- Interpretable output that stakeholders can engage with.

**Limitations:**
- Requires attendance data to discover clusters (same constraint as 3.1).
- Cluster count is a hyperparameter — need to justify K.
- Clusters may not be stable over time as policies change.

**Verdict:** Natural complement to XGBoost attendance prediction. Lower priority than 3.1 but adds narrative value. Good for portfolio storytelling.

---

### 3.3 Sequence Models (HMM / LSTM)

#### 3.3.1 Hidden Markov Model (HMM)

**What it is:** Model each employee's attendance as a sequence of observations generated by hidden states. States might be "office week mode" and "home week mode" with transition probabilities governing how employees shift between modes.

**Strengths:**
- Captures temporal autocorrelation — employees attend in bursts/streaks.
- Mathematically elegant, well-understood inference (Baum-Welch, Viterbi).
- Small number of parameters per employee.

**Limitations:**
- Markov assumption (next state depends only on current state) may be too restrictive.
- Requires per-employee training data — sparse if employees have short tenure.

---

#### 3.3.2 LSTM / Temporal Model

**What it is:** Use an LSTM or simpler RNN to predict tomorrow's attendance based on the sequence of past attendance for each employee.

**Strengths:**
- Can capture long-range dependencies and complex temporal patterns.
- Handles variable-length input sequences.

**Limitations:**
- Requires substantial per-employee history (many months).
- Overkill for most attendance patterns, which are well-captured by day-of-week + short-term autocorrelation.
- Hard to interpret.

**Verdict for both:** HMM is a clean, interpretable option if you want to model attendance streaks. LSTM is hard to justify given the simplicity of the underlying patterns. For the portfolio, HMM > LSTM in terms of defensibility.

---

### 3.4 Team Collaboration Graph Analysis (GNN)

**What it is:** Build a graph where nodes are employees and edges represent collaboration strength (derived from Slack messages, meeting co-attendance, email volume, or code review overlap). Use graph-based methods to identify collaboration clusters and optimize office day scheduling for maximum in-person overlap.

**How it works:**
- Construct the collaboration graph from communication data.
- Apply community detection (Louvain, spectral clustering) or a GNN (GraphSAGE, GCN) to identify tightly-coupled groups.
- Feed these group structures into the optimization layer: "schedule these groups on the same days."
- Alternatively, use the graph to define the "co-location score" objective in multi-objective optimization.

**Strengths:**
- Moves beyond org-chart teams to *actual* collaboration patterns.
- Directly connects to the optimization objective (maximize meaningful co-location).
- Strong portfolio signal — demonstrates graph ML applied to a practical problem.

**Limitations:**
- Requires communication/collaboration data (Slack API, calendar API, etc.).
- Privacy concerns with monitoring communication patterns.
- Graph construction choices (which interactions count? what threshold for an edge?) heavily influence results.

**Verdict:** Excellent stretch goal that differentiates the project. The GNN angle connects directly to your fraud detection GNN experience. For the portfolio, even demonstrating it on synthetic collaboration data is compelling.

---

### 3.5 Anomaly Detection on Attendance Drift

**What it is:** Once the simulator is deployed with calibrated models, monitor whether actual attendance deviates from predictions. Flag when the model's assumptions no longer hold.

**How it works:**
- Each day, compare actual badge-in count against the model's predicted distribution.
- If actual attendance falls outside the 95% prediction interval for multiple consecutive days, flag an alert.
- Potential causes: undocumented policy change, cultural shift, seasonal effect, new team onboarding.
- Triggers model recalibration.

**Approach:** Simple statistical process control (CUSUM, EWMA) or Isolation Forest on the prediction residuals.

**Verdict:** Low-effort, high-value operational addition. Demonstrates production thinking — not just building a model, but monitoring it.

---

## 4. Optimization Approaches

### 4.1 Multi-Objective Optimization (NSGA-II)

**What it is:** Given the simulator as a black-box fitness function, search the space of possible team-day schedules to find the set of solutions that optimally trade off competing objectives.

**Decision variables:** For each team, which days are mandatory office days (combinatorial: team × day assignments).

**Objectives (minimize/maximize):**
- Minimize peak daily occupancy across the month.
- Maximize co-location score (teams that collaborate heavily should share office days).

**Constraints:**
- No day exceeds a capacity threshold (e.g., 90% of seats).
- Every team has at least N mandatory office days per week.
- Optionally respect employee preferences or fixed constraints.

**How NSGA-II works:**
- Maintain a population of candidate schedules.
- Evaluate each candidate by running the simulator (Monte Carlo backend).
- Select, crossover, and mutate to evolve the population.
- Non-dominated sorting identifies the Pareto frontier: the set of solutions where improving one objective necessarily worsens the other.
- Decision-maker picks where on the frontier they want to land.

**Library:** `pymoo` (Python, well-maintained, NSGA-II out of the box).

**Strengths:**
- Produces a Pareto frontier, which is far more useful than a single "optimal" answer — it surfaces the tradeoff explicitly.
- Works with any simulator backend as a black-box evaluator.
- Handles non-linear, non-convex objective landscapes that analytical methods can't.

**Limitations:**
- Computationally expensive: each fitness evaluation requires a full simulation run; population × generations × simulation cost adds up.
- Requires careful objective function design — poor co-location score definition leads to meaningless results.
- Solution quality depends on population size and generations — may need tuning.

**Verdict:** The right tool for the schedule optimization problem. This is the recommended approach for "find me the best team schedule."

---

### 4.2 Bayesian Optimization (Optuna)

**What it is:** A sequential model-based optimization approach that builds a surrogate model (typically Gaussian Process or Tree-Parzen Estimator) of the objective function and intelligently selects the next point to evaluate.

**How it works:**
- Define a composite objective: `score = w1 × peak_occupancy + w2 × (1 - co_location_score)`.
- Optuna proposes a candidate schedule, the simulator evaluates it, Optuna updates its surrogate model, and proposes a better candidate.
- Converges to good solutions with far fewer evaluations than grid search or random search.

**Library:** `Optuna` (Python, excellent API, supports pruning and multi-objective).

**Strengths:**
- Sample-efficient — needs far fewer simulation runs than NSGA-II to find good solutions.
- Simpler to implement than NSGA-II.
- Optuna supports multi-objective (via `optuna.multi_objective`) if needed.

**Limitations:**
- Composite objective requires choosing weights (w1, w2), which is subjective.
- Less suitable for very high-dimensional search spaces (many teams × many days).
- Doesn't naturally produce a Pareto frontier (though Optuna's multi-objective mode does).

**Verdict:** Good starting point before NSGA-II. Easier to implement, faster to converge for moderate search spaces. Use this for v1 optimization, upgrade to NSGA-II if the Pareto frontier visualization is needed.

---

### 4.3 Linear / Integer Programming

**What it is:** Formulate the schedule optimization as a mathematical program with linear objectives and constraints. Solve exactly using an LP/IP solver.

**Formulation sketch:**
- Binary decision variables: `x[team, day] = 1` if team is assigned to office on that day.
- Objective: minimize max daily occupancy (minimax via auxiliary variable).
- Constraints: each team has exactly K office days, capacity limit per day, etc.

**Solver:** `PuLP`, `OR-Tools`, `scipy.optimize.linprog`.

**Strengths:**
- Exact global optimum (no heuristic approximation).
- Very fast for moderate problem sizes.
- Well-understood, mathematically rigorous.

**Limitations:**
- Requires deterministic formulation — loses stochastic richness (uses expected values instead of distributions).
- Linear objectives only (or piecewise-linear approximations).
- Difficult to incorporate complex behavioral constraints.

**Verdict:** Useful as a fast heuristic or to generate a good initial solution that NSGA-II or Bayesian optimization can refine. Not sufficient alone because it ignores stochastic variation.

---

### 4.4 Reinforcement Learning for Policy Search

**What it is:** An RL agent interacts with the simulator as its environment. The state is the current attendance pattern, actions are policy lever adjustments (e.g., change which days are mandatory for which team), and the reward is a composite of utilization and co-location.

**How it works:**
- State: current week's occupancy pattern, team attendance distributions.
- Action: adjust team-day assignments (discrete action space).
- Reward: `r = -α × overflow_count - β × underutilization + γ × co_location_score`.
- Algorithm: PPO or DQN over episodes (each episode = one simulated month).

**Strengths:**
- Can discover non-obvious adaptive policies (e.g., "shift Engineering from Tuesday to Wednesday in high-season months").
- Handles sequential decision-making — policies that evolve over time.

**Limitations:**
- Massive overkill for a static schedule optimization problem.
- Requires careful reward shaping — easy to get degenerate policies.
- Training instability, especially with a stochastic simulator as the environment.
- Very hard to explain to stakeholders why you used RL instead of NSGA-II.

**Verdict:** Interesting academically, but hard to justify practically. The schedule optimization problem doesn't have the sequential, dynamic nature that RL excels at. NSGA-II or Bayesian optimization are better choices. Reserve RL for scenarios where policies need to adapt in real-time to changing conditions.

---

## 5. Approach Comparison Matrix

| Approach | Complexity | Data Needed | Computation Speed | Behavioral Realism | Portfolio Value |
|---|---|---|---|---|---|
| Binomial/Poisson | Very Low | None (params only) | Instant | Low (independent) | Baseline |
| Queueing Theory | Medium | Arrival distributions | Fast | Medium (temporal) | Niche |
| Basic Monte Carlo | Low | None (params only) | Fast (<1s) | Low-Medium | Core |
| Correlated Monte Carlo | Low-Medium | None (params only) | Fast (<1s) | Medium | Recommended |
| SimPy DES | Medium-High | Arrival/departure data | Moderate (seconds) | High (temporal) | Strong |
| Rule-Based ABM (Mesa) | Medium | None (calibrate later) | Moderate (seconds) | High (adaptive) | Strong |
| NN-Brained ABM | High | Training data needed | Slow | Questionable | Avoid |
| System Dynamics | Medium | Feedback strengths | Fast | Macro only | Niche |
| XGBoost Prediction | Medium | Badge data (3-6 months) | Fast (inference) | High (data-driven) | Very Strong |
| Behavioral Clustering | Medium | Badge data | Fast | Medium-High | Good |
| HMM | Medium | Per-employee sequences | Fast | Medium-High | Good |
| LSTM | High | Long sequences needed | Medium | High | Low (overkill) |
| GNN Collaboration | High | Communication data | Medium | High | Very Strong |
| Anomaly Detection | Low | Prediction residuals | Real-time | N/A (monitoring) | Good |
| NSGA-II Optimization | Medium-High | Simulator as evaluator | Slow (minutes) | N/A (optimization) | Very Strong |
| Bayesian Optimization | Medium | Simulator as evaluator | Moderate | N/A (optimization) | Strong |
| Integer Programming | Medium | Expected values | Fast | Low (deterministic) | Moderate |
| RL Policy Search | Very High | Simulator as environment | Very Slow | N/A (optimization) | Low (overkill) |

---

## 6. Recommended Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- **Binomial model** — analytical baseline.
- **Monte Carlo with team correlation** — primary simulation engine.
- Pluggable backend interface.

### Phase 2: UI + Chat (Week 3-4)
- React calendar visualization.
- Claude API integration for natural language query parsing.
- Conversational parameter management.

### Phase 3: Optimization (Week 5-6)
- **Bayesian Optimization** (Optuna) for schedule search.
- Co-location score definition.
- Pareto frontier visualization (upgrade to NSGA-II if needed).

### Phase 4: Advanced Simulation (Week 7-8)
- **SimPy DES backend** — time-of-day dynamics.
- **Mesa ABM backend** — adaptive behavioral dynamics.

### Phase 5: ML Calibration (Week 9-10, if data available)
- **XGBoost attendance prediction** from badge data.
- **Behavioral clustering** (GMM) for employee archetypes.
- **GNN collaboration analysis** for co-location scoring.

### Phase 6: Production Hardening (Stretch)
- **Anomaly detection** on prediction drift.
- Model comparison dashboard (run multiple backends, compare outputs).
- Cost module (translate seat counts to real estate cost).

---

## 7. Key Design Decisions and Rationale

**Why pluggable backends?** The "right" simulation model depends on what data is available and what question is being asked. A facilities team asking "how many seats do we need?" is well-served by Monte Carlo. An HR team asking "how will behavior evolve after a policy change?" needs ABM. The pluggable architecture lets us serve both without architectural changes.

**Why Monte Carlo over Binomial as the primary?** While Binomial gives exact analytical answers for the independence case, Monte Carlo is trivially extensible. Adding team correlation, individual variance, temporal effects, or any new factor requires only modifying the sampling logic — no formula rederivation. The marginal computation cost (sub-second with NumPy vectorization) is negligible.

**Why Bayesian Optimization before NSGA-II?** The schedule search space for 10 teams × 5 days is manageable (~30 binary variables). Bayesian optimization is simpler to implement, converges faster in moderate dimensions, and Optuna's API is excellent. NSGA-II becomes necessary only when the Pareto frontier (explicit tradeoff visualization) is a hard requirement.

**Why rule-based ABM over neural-network ABM?** Simple rules with memory produce the same emergent dynamics that make ABM interesting, without requiring training data for per-agent neural networks. The "AI brain per agent" framing sounds impressive but creates a defensibility problem: you'd be training thousands of tiny NNs on no ground truth. Rule-based ABM with ML-calibrated parameters is more honest and more robust.

**Why XGBoost over deep learning for attendance prediction?** Tabular data (employee attributes + temporal features + environmental factors) is XGBoost's strength. Deep learning shows no consistent advantage on structured tabular problems at this scale. XGBoost also provides feature importance for free, which is critical for stakeholder trust.

---

## 8. References and Libraries

| Library | Purpose | URL |
|---|---|---|
| NumPy / SciPy | Core numerical computation, distributions | numpy.org / scipy.org |
| SimPy | Discrete event simulation | simpy.readthedocs.io |
| Mesa | Agent-based modeling | mesa.readthedocs.io |
| XGBoost | Gradient boosted attendance prediction | xgboost.readthedocs.io |
| LightGBM | Alternative gradient boosting | lightgbm.readthedocs.io |
| scikit-learn | Clustering (GMM, K-Means), preprocessing | scikit-learn.org |
| PyTorch Geometric | GNN for collaboration graphs | pyg.org |
| pymoo | Multi-objective optimization (NSGA-II) | pymoo.org |
| Optuna | Bayesian optimization | optuna.org |
| PuLP / OR-Tools | Linear/integer programming | coin-or.github.io/pulp / developers.google.com/optimization |
| Streamlit / React | Frontend dashboard | streamlit.io / react.dev |
| FastAPI | Backend API | fastapi.tiangolo.com |
