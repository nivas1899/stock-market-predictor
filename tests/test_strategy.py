import itertools
import numpy as np
import pytest
from market_predictor.strategy import plan_strategy


def exhaustive(prices, fee, budget):
    best = 1.0
    for path in itertools.product((0, 1), repeat=len(prices)-1):
        previous, orders, value = 0, 0, 1.0
        for i, held in enumerate(path):
            switch = held != previous
            orders += int(switch)
            value *= (1-fee)**int(switch) * (prices[i+1]/prices[i])**held
            previous = held
        orders += previous
        value *= (1-fee)**previous
        if orders <= budget:
            best = max(best, value)
    return best


@pytest.mark.parametrize("fee", [0, .001, .03])
@pytest.mark.parametrize("budget", [0, 1, 2, 4, 8])
def test_astar_matches_exhaustive(fee, budget):
    rng = np.random.default_rng(123)
    for _ in range(10):
        prices = np.exp(rng.normal(0, .1, 7)) * 100
        plan = plan_strategy(prices, 10000, fee, budget)
        assert plan.final_value == pytest.approx(10000 * exhaustive(prices, fee, budget))
        assert plan.orders <= budget
        assert len(plan.ledger) == len(prices)-1


def test_rising_path_buys_once_and_liquidates():
    plan = plan_strategy([100, 110, 120], fee=.01, max_orders=2)
    assert plan.orders == 2
    assert plan.final_value == pytest.approx(10000 * 1.2 * .99**2)
    assert plan.ledger.Action.tolist() == ["BUY", "HOLD"]
    assert plan.ledger["Liquidate at end"].iloc[-1]


@pytest.mark.parametrize("prices", [[100], [100, 0], [100, np.nan], [100, np.inf]])
def test_invalid_prices(prices):
    with pytest.raises(ValueError):
        plan_strategy(prices)
