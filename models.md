# Simulation Models

The RTO Capacity Simulator provides three interchangeable simulation backends, each registered under a unique id in the backend registry. All three share the same input/output schema (`SimulationInput` → `SimulationResult`) and are selectable at runtime via the API or Streamlit sidebar.

---

## Common Inputs

Every model receives a `SimulationInput` with the following fields:

| Field | Default | Description |
|---|---|---|
| `total_employees` | — | Total headcount |
| `total_seats` | — | Physical seats available |
| `wfh_days_per_week` | 2 | Days per week each employee works from home |
| `seat_reduction_pct` | 0.0 | % of seats removed (social distancing, hot-desking reserve, etc.) |
| `mandatory_office_days` | `{}` | Per-team forced in-office days, e.g. `{"Engineering": ["Tuesday","Wednesday"]}` |
| `day_of_week_weights` | Mon 0.85 … Fri 0.65 | Relative likelihood each day of week is chosen as an office day |
| `compliance_rate` | 0.9 | Fraction of employees who actually follow their assigned schedule |
| `teams` | `[]` | Optional list of `{"name": str, "size": int}` sub-groups |
| `num_simulation_runs` | 5000 | Replications (Monte Carlo and DES only) |
| `start_date` / `end_date` | current month | Date range to simulate |

### Effective attendance probability

All three models share the same formula to compute `p_effective` — the probability that a given employee comes into the office on a given day:

```
# If the day is a mandatory office day for the employee's team:
p_effective = compliance_rate

# Otherwise:
base_p = (5 - wfh_days_per_week) / 5
p_effective = base_p × day_of_week_weight × compliance_rate
```

### Effective seat capacity

```
effective_capacity = floor(total_seats × (1 - seat_reduction_pct / 100))
```

---

## Model 1 — Binomial / Analytical

**Registry id:** `binomial`
**File:** [backend/binomial_backend.py](backend/binomial_backend.py)

### How it works

This is a closed-form analytical model. No random sampling is performed. For each working day, it computes the exact mean and variance of attendance using the Binomial distribution, then uses a Normal approximation to derive overflow probability and confidence intervals.

**Per day, per team:**

```
mean     = n × p_effective
variance = n × p_effective × (1 − p_effective)
```

Multiple teams are treated as independent, so totals are summed:

```
total_mean     = Σ mean_i
total_variance = Σ variance_i
total_std_dev  = √total_variance
```

**Overflow probability** (Normal approximation of Binomial):

```
z = (effective_capacity − total_mean) / total_std_dev
P(overflow) = 1 − Φ(z)
```

where Φ is the standard Normal CDF.

**Confidence interval** (5th / 95th percentile from Normal):

```
p5  = Φ⁻¹(0.05, μ=total_mean, σ=total_std_dev)
p95 = Φ⁻¹(0.95, μ=total_mean, σ=total_std_dev)
```

### Assumptions and limitations

- **Independence:** employees make their attendance decisions independently. Team-level social effects (e.g. "if my team is going in, I'll go too") are not modelled.
- **Normal approximation:** accurate when `n` is large (>30) and `p` is not near 0 or 1.
- **Daily totals only:** the model has no concept of time within a day. It does not capture peak-hour crowding.

### When to use

Best for quick, deterministic estimates where speed matters and the assumptions of independence are acceptable. Results are reproducible (no randomness) and arrive instantly regardless of date range.

---

## Model 2 — Monte Carlo

**Registry id:** `monte_carlo`
**File:** [backend/monte_carlo_backend.py](backend/monte_carlo_backend.py)

### How it works

The Monte Carlo backend runs `num_simulation_runs` stochastic replications of each working day. Rather than computing exact probabilities, it draws random samples and derives statistics empirically from the distribution of outcomes.

**Two sources of randomness beyond `p_effective`:**

1. **Team-level social factor** — captures the tendency for team members to cluster their office days together. Drawn once per team per run from a Beta(8, 2) distribution, normalised to mean 1.0:
   ```
   social_factor ~ Beta(8, 2) / (8/10)   # mean=1.0, low variance
   ```

2. **Individual reliability factor** — captures personal variation in schedule adherence (illness, travel, etc.). Drawn once per employee per team from a Beta(5, 1) distribution, normalised to mean 1.0:
   ```
   reliability ~ Beta(5, 1) / (5/6)   # mean=1.0, right-skewed
   ```

**Effective probability per employee per run:**

```
p_matrix[run, employee] = clip(p_effective × social_factor[run] × reliability[run, employee], 0, 1)
```

**Attendance draw** (Bernoulli per employee per run):

```
attendance[run, employee] = Uniform(0,1) < p_matrix[run, employee]
daily_count[run] = Σ attendance[run, :]
```

**Statistics** derived empirically across all runs:

```
expected_occupancy  = mean(daily_count)
std_dev             = std(daily_count)
overflow_probability = count(daily_count > capacity) / n_runs
p5, p95             = percentile(daily_count, [5, 95])
```

### Assumptions and limitations

- **Team correlation, not cross-team:** the social factor is applied independently per team. Cross-team correlations are not modelled.
- **No intraday timing:** the model produces a daily headcount, not a time-series. It does not distinguish between everyone arriving at 9am vs. spread across the day.
- **Variance normalisation:** Beta distributions are normalised to mean 1.0 so neither factor biases the mean occupancy — they add realistic variance only.

### When to use

Best when you need realistic occupancy distributions rather than point estimates, especially for scenarios with team-based attendance clustering or variable individual reliability. The empirical percentiles give a more accurate uncertainty range than the Normal approximation under non-Gaussian conditions. Slower than Binomial but still fast (~2s for 5000 runs × 1 month).

---

## Model 3 — SimPy Discrete-Event Simulation

**Registry id:** `simpy_des`
**File:** [backend/simpy_backend.py](backend/simpy_backend.py)

### How it works

The DES backend models intraday office dynamics at the individual event level. Each employee is an agent who:

1. **Decides** whether to come in (same `p_effective` logic as above, via Binomial draw)
2. **Arrives** at a random time drawn from Normal(μ=9:00am, σ=1h), clipped to 7am–12pm
3. **Claims a seat** on a first-come-first-served basis
4. **Works** for a random duration drawn from Normal(μ=8h, σ=1h), clipped to 4h–10h
5. **Departs**, freeing their seat

If no seat is available when an employee arrives, they are **turned away** immediately (no waiting). Overflow is recorded for that run.

**Implementation — heapq sweep-line scheduler:**

Rather than spawning one Python coroutine per employee (which proved ~50× too slow), the inner simulation uses a min-heap to process events in arrival-time order:

```
arrivals  ~ Normal(9×60, 60) minutes, clipped to [7×60, 12×60]
durations ~ Normal(8×60, 60) minutes, clipped to [4×60, 10×60]
departures = arrivals + durations

Sort employees by arrival time.
heap = []   # departure times of currently occupied seats

for each employee in arrival order:
    pop from heap all departure times ≤ arrival_time  # free vacant seats
    if len(heap) < seats:
        push departure_time onto heap                  # employee takes a seat
        update peak_occupancy = max(peak, len(heap))
    else:
        turned_away += 1                               # no seat available
```

**Per day, across `num_simulation_runs` replications:**

```
peaks[run]         = peak_concurrent_occupancy for that run
turned_away[run]   = employees who could not get a seat

expected_occupancy  = mean(peaks)
overflow_probability = count(turned_away > 0) / n_runs
p5, p95             = percentile(peaks, [5, 95])
```

**Key difference from other models:**
`expected_occupancy` here is **peak concurrent occupancy** (bounded by seat count), not total demand. A day where 350 employees want to come in but only 300 seats exist will show `expected_occupancy ≈ 300` and `overflow_probability > 0`. The other two models would show `expected_occupancy ≈ 350`.

### Assumptions and limitations

- **No queuing / waiting:** employees who find no seat leave immediately (balking behaviour). No waiting room or hot-desk queue is modelled.
- **Arrival window fixed:** arrivals are clipped to 7am–12pm. Afternoon arrivals are not modelled.
- **No team social factor:** the DES uses independent Binomial draws for attendance, without the Beta-distributed social/reliability factors of Monte Carlo.
- **Peak, not total:** `expected_occupancy` reflects peak concurrent, not total daily footfall. This can appear lower than Binomial/MC for the same inputs when seat contention is low.

### When to use

Best when intraday timing matters — for example, when evaluating staggered start times, or when peak-hour crowding is the actual risk rather than total headcount. Also useful for validating that a physical seat count is sufficient given realistic arrival patterns, not just daily averages.

---

## Comparison at a Glance

| | Binomial | Monte Carlo | SimPy DES |
|---|---|---|---|
| **Approach** | Closed-form analytical | Stochastic sampling | Agent-based DES |
| **Randomness** | None | Team social + individual reliability | Arrival time + work duration |
| **Overflow detection** | `mean > capacity` (expected demand) | `sample > capacity` (empirical) | `turned_away > 0` (physical contention) |
| **`expected_occupancy` meaning** | Mean employees wanting to come in | Mean employees wanting to come in | Mean peak concurrent in seats |
| **Intraday timing** | No | No | Yes |
| **Team correlation** | No | Yes (social factor) | No |
| **Speed** | Instant | ~2s / 5000 runs / month | ~9s / 1000 runs / month |
| **Best for** | Fast estimates, reproducibility | Realistic distributions, team effects | Peak-hour crowding, seat contention |
| **Cross-backend agreement** | Reference | Within ±10% of Binomial | Within ±15% of Binomial |

---

## Output Fields (all models)

Each model returns a `SimulationResult` containing:

- **`daily_results`** — one `DayResult` per working day:
  - `expected_occupancy` — mean occupancy (interpretation differs by model, see above)
  - `std_dev` — standard deviation across runs (or from Normal approximation)
  - `overflow_probability` — probability of exceeding seat capacity
  - `percentile_5`, `percentile_95` — confidence interval bounds
  - `effective_capacity` — seats available that day (after reduction)
  - `team_breakdown` — expected headcount per team

- **`summary`**:
  - `avg_utilization` — mean occupancy / capacity across all days
  - `peak_occupancy` — highest single-day expected occupancy
  - `overflow_days_count` — number of days with overflow risk
  - `overflow_days_pct` — fraction of working days with overflow risk
  - `avg_overflow_magnitude` — average excess demand on overflow days
