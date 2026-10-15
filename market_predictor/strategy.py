"""Exact A* search over flat/long exposure with fees and an order budget."""
from dataclasses import dataclass
import heapq
import itertools
import math
import numpy as np
import pandas as pd


@dataclass
class Plan:
    ledger: pd.DataFrame
    final_value: float
    expanded: int
    orders: int


def plan_strategy(prices, capital=10000.0, fee=0.001, max_orders=8) -> Plan:
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or not 2 <= len(prices) <= 61:
        raise ValueError("A* expects 2 to 61 prices, including the starting close.")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Strategy prices must be finite and positive.")
    if not math.isfinite(capital) or capital <= 0 or not 0 <= fee < 1:
        raise ValueError("Capital must be positive and fee must be in [0, 1).")
    if not isinstance(max_orders, int) or not 0 <= max_orders <= 60:
        raise ValueError("Order budget must be an integer between 0 and 60.")
    log_returns = np.log(prices[1:] / prices[:-1])
    # Subtract actual log growth from an upper bound, yielding nonnegative edges.
    bounds = np.abs(log_returns) + 1.0
    # Relaxation: switch exposure freely at no cost on each remaining interval.
    relaxed = bounds - np.maximum(log_returns, 0)
    heuristic = np.r_[np.cumsum(relaxed[::-1])[::-1], 0.0]
    start = (0, 0, 0)  # interval, exposure (0=cash, 1=long), orders used
    serial = itertools.count()
    queue = [(float(heuristic[0]), next(serial), 0.0, start)]
    best, parent = {start: 0.0}, {}
    expanded = 0
    terminal = None
    while queue:
        _, _, cost, state = heapq.heappop(queue)
        if cost > best.get(state, math.inf) + 1e-12:
            continue
        day, held, orders = state
        expanded += 1
        if day == len(log_returns):
            terminal = state
            break
        for exposure in (0, 1):
            switching = int(exposure != held)
            liquidate = int(day == len(log_returns)-1 and exposure == 1)
            used = orders + switching + liquidate
            if used > max_orders:
                continue
            growth = exposure * log_returns[day] + (switching+liquidate) * math.log1p(-fee)
            next_state = (day+1, 0 if liquidate else exposure, used)
            new_cost = cost + bounds[day] - growth
            if new_cost < best.get(next_state, math.inf) - 1e-12:
                best[next_state] = new_cost
                parent[next_state] = (state, exposure, switching, liquidate, growth)
                heapq.heappush(queue, (new_cost + heuristic[day+1], next(serial),
                                       new_cost, next_state))
    path = []
    state = terminal
    while state != start:
        previous, exposure, switch, liquidate, growth = parent[state]
        path.append((previous[0], exposure, switch, liquidate, growth))
        state = previous
    value, rows = capital, []
    for day, exposure, switch, liquidate, growth in reversed(path):
        value *= math.exp(growth)
        action = ("BUY" if exposure else "SELL") if switch else "HOLD" if exposure else "CASH"
        rows.append({"Step": day, "Price": prices[day], "Next price": prices[day+1],
                     "Action": action, "Exposure": exposure,
                     "Liquidate at end": bool(liquidate), "Equity": value})
    return Plan(pd.DataFrame(rows), value, expanded, terminal[2])


def buy_and_hold(prices, capital, fee):
    prices = np.asarray(prices, dtype=float)
    return capital * (1-fee)**2 * prices[-1] / prices[0]
