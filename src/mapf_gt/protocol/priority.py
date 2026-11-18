"""Priority and ageing rules, shared by ``BiddingAgent`` and ``UrgencyPIBT``.

Keeping one copy means the decentralised protocol and the centralised oracle
can never disagree about how a priority is built.

The default configuration is rule E (urgency-weighted ageing):

    priority = (tier, b * (d + alpha * age), id)
    age      = timesteps since the agent was last at its goal (0 at the goal)

The switches in ``ProtocolConfig`` reproduce the alternatives compared in
``experiments/compare_rules.py``:

* ``priority_rule``: "weighted" (E) gives b * (d + alpha * age);
  "additive" gives b * d + alpha * age.
* ``age_rule``: "since_goal" (E) adds 1 every tick away from the goal;
  "since_progress" (the plan's original rule) also resets to 0 whenever the
  agent gets closer to its goal.
* ``emergency_counter``: which counter the emergency tier checks against
  ``emergency_wait``: "age" (the priority counter) or "since_goal".
"""

from __future__ import annotations

from .context import Priority

PRIORITY_RULES = ("weighted", "additive")
AGE_RULES = ("since_goal", "since_progress")
EMERGENCY_COUNTERS = ("age", "since_goal")


def priority(
    bid: float, delay: float, age: int, since_goal: int, agent_id: int, config
) -> Priority:
    """(tier, value, id) for one agent this tick."""
    counter = age if config.emergency_counter == "age" else since_goal
    emergency = config.emergency_wait is not None and counter >= config.emergency_wait
    if config.priority_rule == "weighted":
        value = bid * (delay + config.alpha * age)
    elif config.priority_rule == "additive":
        value = bid * delay + config.alpha * age
    else:
        raise ValueError(f"priority_rule must be one of {PRIORITY_RULES}")
    return (1 if emergency else 0, value, agent_id)


def next_age(age: int, at_goal: bool, got_closer: bool, config) -> int:
    """The priority counter after one tick."""
    if config.age_rule == "since_goal":
        return 0 if at_goal else age + 1
    if config.age_rule == "since_progress":
        return 0 if at_goal or got_closer else age + 1
    raise ValueError(f"age_rule must be one of {AGE_RULES}")


def next_since_goal(since_goal: int, at_goal: bool) -> int:
    """Timesteps since last at goal, whatever ``age_rule`` is."""
    return 0 if at_goal else since_goal + 1
