from collections import deque
from decimal import Decimal, getcontext
import math
import pandas as pd
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

getcontext().prec = 28  # high precision for Decimal arithmetic

# --- 1. Position class (provided) ---
class Position:
    """Tracks current position quantity, average cost (VWAP), and realized PnL."""
    def __init__(self) -> None:
        self._qty = Decimal("0")
        self._realised_pnl = Decimal("0")
        self._lots = deque()          # FIFO lots: each {"qty": Decimal, "price": Decimal}
        self._ids_seen = set()        # to avoid double-processing same exec ID

    def apply_fill(self, side: str, qty: float, price: float, fee: float = 0.0, exec_id: str = None) -> None:
        # Idempotency check
        if exec_id and exec_id in self._ids_seen:
            return
        if exec_id:
            self._ids_seen.add(exec_id)

        q = Decimal(str(qty))
        p = Decimal(str(price))
        f = Decimal(str(fee))
        if side.lower() == "buy":
            # Add lot: incorporate fee into effective price
            effective_price = (p * q + f) / q
            self._lots.append({"qty": q, "price": effective_price})
            self._qty += q
        elif side.lower() == "sell":
            if q > self._qty:
                raise ValueError("Sell qty exceeds current position")
            remaining = q
            cost_basis = Decimal("0")
            # Deplete from FIFO lots to compute PnL
            while remaining > 0:
                lot = self._lots[0]
                take_qty = min(lot["qty"], remaining)
                cost_basis += take_qty * lot["price"]
                lot["qty"] -= take_qty
                remaining -= take_qty
                if lot["qty"] == 0:
                    self._lots.popleft()
            proceeds = q * p
            tx_fee = (proceeds + cost_basis) * Decimal("0.003")
            self._realised_pnl += proceeds - cost_basis - tx_fee
            self._qty -= q
        else:
            raise ValueError("Side must be 'buy' or 'sell'")

    @property
    def qty(self) -> Decimal:
        return self._qty

    @property
    def avg_cost(self) -> Decimal:
        """Current volume-weighted average cost of the position."""
        if self._qty == 0:
            return Decimal("0")
        total_cost = sum(lot["qty"] * lot["price"] for lot in self._lots)
        return total_cost / self._qty

    @property
    def realised_pnl(self) -> Decimal:
        return self._realised_pnl

    def is_flat(self) -> bool:
        """Returns True if position is zero (no holdings)."""
        return self._qty == 0

# --- 2. Strategy state class (provided, with an added flag for skip logic) ---
class StrategyState:
    def __init__(self):
        # Track lowest price seen since position opened (for drawdown calculations)
        self.min_price: float = None
        # Average cost before the most recent buy (for reference)
        self.prior_avg_cost: float = None
        # Planned number of partial sell orders for current position (set during sell logic)
        self.num_remaining_order: int = None
        # Last sell execution price and timestamp (for cooldown logic)
        self.last_sell_price: float = None
        self.last_sell_ts: float = None
        # Highest price observed since last sell (for cooldown drop trigger)
        self.highest_price_since_last_sell: float = None
        # Flag for price has draw down significantly from peak since last sell, for cooldown drop trigger
        self.price_big_down_since_last_sell: bool = False
        # Flag for initial profit threshold (before first sell in a cycle)
        self.initial_threshold_active: bool = True
        # Counter of how many buy operations have been performed in the current position
        self.buy_count: int = 0
        # Flag to enforce skipping one large-sell signal after the second buy (for rule 5.3)
        self.skip_after_second: bool = False

# --- 3. Sell decision function (provided) ---
def sell(tx_row: pd.Series, pos: Position, state: StrategyState):
    """
    Decide whether to sell a portion of the position based on a large buy order (price up).
    Returns a dict with keys: sell (bool), sell_qty, sell_price, reason, cooldown.
    """
    decision = {"sell": False, "sell_qty": 0.0, "sell_price": None, "reason": "", "cooldown": False}
    # If no position, skip selling
    if pos.is_flat():
        decision["reason"] = "No position to sell"
        return decision

    # Calculate current market price after this swap.
    # For a buy swap (user buys base, price goes up), final price = (Q_before + amt_in) / (B_before - amt_out).
    current_price = (tx_row["amt_in"] + tx_row["pool_quote"]) / (tx_row["pool_base"] - tx_row["amt_out"])

    state.price_big_down_since_last_sell = (current_price <= state.highest_price_since_last_sell * 0.90) if state.highest_price_since_last_sell else False  # price dropped 10% from post-sell peak

    # Update minimum price seen since position opened (for drawdown tracking)
    if state.min_price is None:
        state.min_price = current_price
    else:
        state.min_price = min(state.min_price, current_price)

    # If still in initial profit phase, allow an override if a ≥7% drawdown from avg cost occurred
    if state.initial_threshold_active:
        if state.min_price <= float(pos.avg_cost) * (1 - 0.07):
            state.initial_threshold_active = False  # drawdown override: lift profit threshold requirement

    # Calculate this buy swap's price impact (amt_out removed from pool_base)
    market_impact = 0.0
    if tx_row["pool_base"] != 0:
        market_impact = tx_row["amt_out"] / tx_row["pool_base"]

    # **Sell Rule 1:** Only consider selling if price impact is above a minimum (e.g. ≥0.9%)
    MIN_IMPACT_THRESHOLD = 0.009  # 0.9% of pool base
    if market_impact < MIN_IMPACT_THRESHOLD:
        decision["reason"] = "Impact below threshold, no sell"
        # If a sell just happened recently, mark cooldown (still in enforced waiting period)
        if state.last_sell_ts is not None:
            decision["cooldown"] = True
        return decision

    # **Sell Rule 2:** If initial profit threshold is active, require 5% profit from avg cost
    if state.initial_threshold_active:
        target_price = float(pos.avg_cost) * 1.05
        if current_price < target_price:
            decision["reason"] = "Initial profit threshold not met"
            return decision
        # If we reach here, profit threshold met; we'll proceed to sell and disable the threshold hereafter.

    # **Sell Rule 9:** If in drawdown from prior average cost, consider selling only when has bounced 30% from min_price
    if state.prior_avg_cost is not None and (current_price - state.min_price) <= (float(state.prior_avg_cost) - state.min_price) * 0.3:
        decision['reason'] = "Still at the price bottom"
        return decision

    # **Sell Cooldown:** Enforce cooldown after a sell unless certain conditions are met
    if state.last_sell_ts is not None:
        # Track highest price since last sell
        if state.highest_price_since_last_sell is None:
            state.highest_price_since_last_sell = current_price
        else:
            state.highest_price_since_last_sell = max(state.highest_price_since_last_sell, current_price)
        # Time since last sell (assuming timestamps are numeric or comparable)
        now_ts = tx_row.get("ts", None)
        time_since_last = (now_ts - state.last_sell_ts) if (now_ts is not None and state.last_sell_ts is not None) else float('inf')
        # Cooldown conditions to allow a new sell:
        price_up = (current_price >= state.last_sell_price * 1.04) if state.last_sell_price else False   # price went 4% above last sell
        time_passed = time_since_last >= 7  # at least 7 seconds (or units) since last sell
        if not (price_up or state.price_big_down_since_last_sell or time_passed):
            # Still in cooldown – skip selling
            decision["reason"] = "Cooldown active"
            decision["cooldown"] = True
            return decision

    # **If we reach here, conditions to sell are satisfied.**

    # Determine how many partial sells to split into based on position size vs pool liquidity
    baseline_impact = 0.003  # we aim for ~0.3% impact per sell
    position_impact = float(pos.qty) / tx_row["pool_base"] if tx_row["pool_base"] != 0 else float('inf')
    # Plan number of sells as the ceiling of position impact / baseline impact (at least 1)
    num_orders = max(1, math.ceil(position_impact / baseline_impact))
    state.num_remaining_order = num_orders  # total planned sells in this cycle

    # Determine fraction of position to sell now.
    baseline_portion = 0.2  # base portion for one unit of impact scaling
    baseline_market_imp = 0.015  # 1.5% impact corresponds to baseline_portion
    # Scale up sell portion if the market impact is higher (sell more if the pump is big)
    scale_factor = math.floor(market_impact / baseline_market_imp)
    sell_portion = max(1.0/num_orders, baseline_portion * scale_factor)
    sell_portion = min(sell_portion, 1.0)  # cannot exceed 100% of position

    # **Sell Rule 3:** If >10% drawdown happened and price nearly recovered (≥99% of avg cost), unload most of position
    drawdown_override = False
    if state.min_price and float(state.min_price) < float(pos.avg_cost) * 0.90:
        if current_price >= float(pos.avg_cost) * 0.99:
            if sell_portion < 0.90:
                sell_portion = 0.90  # sell at least 90% to reduce risk
            drawdown_override = True

    # Calculate sell quantity and price
    total_qty = float(pos.qty)
    sell_qty = min(total_qty, math.ceil(total_qty * sell_portion))
    if sell_portion >= 0.99:  # if nearly whole position, just sell all to close
        sell_qty = total_qty
    sell_price = current_price  # assume execution at current price

    # Sandwich attack detection: if price impact > 4%, defer sell decision to await confirmation
    if market_impact > 0.04:
        decision["sell"] = False
        decision["defer_sell"] = True
        decision["sell_qty"] = sell_qty
        decision["sell_price"] = sell_price
        decision["reason"] = "Sandwich attack suspected: delaying sell"
        decision["cooldown"] = False
        return decision

    decision["sell"] = True

    # Update state for this sell execution
    remaining_qty = total_qty - sell_qty
    if remaining_qty > 0:
        # Recompute planned sells for remaining position (if position not fully closed)
        new_impact = (remaining_qty / tx_row["pool_base"]) if tx_row["pool_base"] != 0 else float('inf')
        state.num_remaining_order = max(1, math.ceil(new_impact / baseline_impact))
    else:
        state.num_remaining_order = 0

    state.last_sell_price = sell_price
    state.last_sell_ts = tx_row.get("ts", None)
    state.highest_price_since_last_sell = current_price  # reset peak tracking after sell
    # Deactivate initial profit threshold after first sell
    state.initial_threshold_active = False

    # If position closed, reset state for next cycle
    if remaining_qty <= 0:
        state.min_price = None
        state.prior_avg_cost = None
        state.num_remaining_order = None
        state.last_sell_price = None
        state.last_sell_ts = None
        state.highest_price_since_last_sell = None
        state.initial_threshold_active = True
        state.buy_count = 0
        state.skip_after_second = False

    # Reason/message for logging or debugging
    if drawdown_override:
        decision["reason"] = f"Drawdown recovery: selling {sell_portion*100:.0f}% of position"
    elif state.buy_count <= 1 and sell_portion >= 1.0:
        decision["reason"] = "Initial target achieved: closing position"
    elif state.buy_count <= 1:
        decision["reason"] = f"Initial profit target reached, selling {sell_portion*100:.0f}%"
    else:
        decision["reason"] = f"Scaling out: sold {sell_portion*100:.0f}% of position"

    decision["sell_qty"] = sell_qty
    decision["sell_price"] = sell_price
    decision["cooldown"] = False  # we are executing a sell now (cooldown enforced after this)
    return decision

# --- 4. Buy decision function (to be implemented according to rules) ---
# Constants for buy impact calculation
BUY_THRESHOLD = 0.02   # 2% price impact threshold for large sell
IMPACT_K = 0.55        # parameters for impact_multiplier formula
IMPACT_CAP = 2.0
POS_FRACTION = 0.16

def impact_multiplier(impact: float) -> float:
    """
    Calculate a size multiplier based on how large the impact is relative to threshold.
    Uses a log scale: m = 1 + K * log(impact/threshold), capped between 1 and IMPACT_CAP.
    """
    if impact <= 0:
        return 1.0
    ratio = impact / BUY_THRESHOLD
    m = 1 + IMPACT_K * math.log(ratio)
    # Ensure multiplier is at least 1 and at most IMPACT_CAP
    return min(IMPACT_CAP, max(1.0, m))

def buy(tx_row: pd.Series, pos: Position, state: StrategyState, feed: pd.DataFrame, idx: int):
    """
    Decide whether to buy in response to a large sell swap.
    Returns a dict with keys: buy (bool), buy_qty, buy_price, reason.
    """
    decision = {"buy": False, "buy_qty": 0.0, "buy_price": None, "reason": ""}

    # Compute current pool price after the sell, to use as our buy price.
    # After a sell: base increases by amt_in, quote decreases by amt_out.
    price_after_sell = (tx_row["pool_quote"] - tx_row["amt_out"]) / (tx_row["pool_base"] + tx_row["amt_in"]) if tx_row["pool_base"] and tx_row["pool_quote"] else None
    if price_after_sell is None:
        # Fallback: use average price of this swap as an approximation (amt_out/amt_in)
        price_after_sell = tx_row["amt_out"] / tx_row["amt_in"] if tx_row["amt_in"] != 0 else 0.0

    current_price = float(price_after_sell)
    state.price_big_down_since_last_sell = (current_price <= state.highest_price_since_last_sell * 0.90) if state.highest_price_since_last_sell else False  # price dropped 10% from post-sell peak

    if state.buy_count > 0:
        if state.min_price is None:
            state.min_price = current_price
        else:
            state.min_price = min(state.min_price, current_price)

    if state.last_sell_ts is not None:
        # Track highest price since last sell
        if state.highest_price_since_last_sell is None:
            state.highest_price_since_last_sell = current_price
        else:
            state.highest_price_since_last_sell = max(state.highest_price_since_last_sell, current_price)

    # Only act on sell swaps (user selling base into pool causing price drop)
    if tx_row["direction"] != "sell":
        decision["reason"] = "Not a sell swap"
        return decision

    # Calculate price impact of this sell: amt_in (base added) as fraction of pool_base.
    # (Assume tx_row["pool_base"] reflects pool base *before* the swap for impact calculation)
    pool_base_before = float(tx_row["pool_base"])
    impact = tx_row["amt_in"] / pool_base_before if pool_base_before > 0 else 0.0

    # **Buy Rule 2:** Require impact >= threshold (e.g. ≥2% of pool) to trigger a buy.
    if impact < BUY_THRESHOLD:
        decision["reason"] = "Sell order impact below threshold"
        return decision

    # Only gate the *first* buy in a cycle
    if state.buy_count == 0:
        macd_now = tx_row["macd"]
        macd_prev = feed.iloc[idx-1]["macd"] if idx > 0 else float("nan")
        if idx == 0 or pd.isna(macd_prev) or macd_now <= macd_prev:
            decision["reason"] = "MACD not rising, skip initial buy"
            return decision 

    # **Buy Rule 3:** Sandwich attack check – look two swaps back for attacker’s buy.
    if idx >= 2:
        prev2_swap = feed.iloc[idx - 2]
        if (prev2_swap["direction"] == "buy" and
            prev2_swap["initiator"] == tx_row["initiator"] and
            abs(prev2_swap["amt_out"] - tx_row["amt_in"]) < 1e-9):
            # Same initiator did a buy of the same amount just before the previous swap -> likely sandwich
            decision["reason"] = "Sandwich attack suspected: skip buy"
            return decision

    # **Buy Rule 5.1:** Limit total number of buys in this cycle to 3.
    if state.buy_count >= 3:
        decision["reason"] = "Max buy count reached for this cycle"
        return decision

    # **Buy Rule 5.2:** If this is an additional buy (not the first), require price at least 10% below current avg cost.
    if state.buy_count > 0:
        if current_price > float(pos.avg_cost) * 0.90:  # price not low enough relative to break-even
            decision["reason"] = "Price not 10% below avg cost, skip buy"
            return decision
        # **Buy Rule 5.3:** If considering the 3rd buy, enforce skipping one large-sell signal after the 2nd buy.
        if state.buy_count == 2 and state.skip_after_second:
            # Skip this signal (the first one after second buy), then clear the flag
            decision["reason"] = "Skipped one large sell after 2nd buy (per strategy rule)"
            state.skip_after_second = False
            return decision

    # **All conditions passed – execute a buy.**
    decision["buy"] = True
    # Determine buy quantity = amt_in (size of dump) * multiplier (based on impact severity)
    mult = impact_multiplier(impact)
    buy_qty = math.floor(tx_row["amt_in"] * mult * POS_FRACTION)

    # Cap buy_qty by 1% of pool_base
    pool_base_cap = round(tx_row["pool_base"] * 0.005 * mult)
    buy_qty = min(buy_qty, pool_base_cap)

    buy_price = current_price

    # Update strategy state before executing buy
    state.prior_avg_cost = float(pos.avg_cost) if pos and pos.qty > 0 else None
    state.buy_count += 1
    if state.buy_count == 2:
        # After executing the 2nd buy, set flag to skip the next large-sell signal
        state.skip_after_second = True

    decision["impact"] = impact
    decision["buy_qty"] = buy_qty
    decision["buy_price"] = buy_price
    decision["reason"] = f"Buy triggered: large sell impact {impact*100:.1f}%, buying {buy_qty:.0f} at ${buy_price:.6f}"
    return decision

def compute_post_price(row):
    if row["direction"] == "sell":
        return (row["pool_quote"] - row["amt_out"]) / (row["pool_base"] + row["amt_in"])
    else:  # direction == "buy"
        return (row["amt_in"] + row["pool_quote"]) / (row["pool_base"] - row["amt_out"])

# ------------------------------------------------------------------ #
# 5. Back‑test loop
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Load transaction feed CSVs
    import os, glob
    os.makedirs("backtest_result-2", exist_ok=True)
    strategy_pnls = []
    target_pnls = []
    strategy_impacts = []
    strategy_buys_counts = []
    target_buys_counts = []
    strat_cycle_sell_counts = []
    target_cycle_sell_counts = []
    for file_path in glob.glob("tx_record2/*.csv"):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        feed_df = pd.read_csv(file_path)
        feed_df['price'] = float('nan')
        feed_df['price_impact'] = float('nan')
        feed_df["trade_triggered"] = ""        # empty string for no trigger
        feed_df["trade_executed"] = ""         # empty string for no execution
        feed_df["trade_size"] = float('nan')   # NaN for no trade size
        feed_df["trade_price"] = float('nan')  # NaN for no trade price
        feed_df["position"] = float('nan')     # NaN for no position change

        feed_df["price"] = feed_df.apply(compute_post_price, axis=1)
        #### Build a complete slot index so empty slots can be forward-filled later
        all_slots = range(int(feed_df["slot"].min()), int(feed_df["slot"].max()) + 1)

        # Take the LAST price in each slot → the slot’s close price
        slot_close = (
            feed_df.groupby("slot")["price"]
                .last()               # last swap in that slot
                .reindex(all_slots)   # include slots with no swaps
                .ffill()              # forward-fill empty slots
        )
        FAST, SLOW = 1200, 2600
        ema_fast = slot_close.ewm(span=FAST, adjust=False).mean()
        ema_slow = slot_close.ewm(span=SLOW, adjust=False).mean()
        macd      = ema_fast - ema_slow
        macd_next_slot = macd.shift(1)
        slot2ts = pd.to_datetime(
            feed_df.groupby("slot")["ts"].first().reindex(all_slots).ffill().astype(int),
            unit="s", utc=True
        )
        macd_ts = slot2ts.reindex(macd_next_slot.index)  # ensure same length/order
        macd_x = macd_ts.values          # datetime64[ns] values
        macd_y = macd_next_slot.values   # MACD values already shifted

        #### attach to each swap row (slot→MACD) for use in buy-logic
        feed_df["macd"] = feed_df["slot"].map(macd_next_slot)

        pos = Position()
        state = StrategyState()
        b_delay = 2
        s_delay = 1

        # Lists to track performance over time
        pnl_history = []       # cumulative realized PnL after each swap
        position_history = []  # position size after each swap (for reference)
        volume_history = []       # cumulative trading volume (quote currency)
        position_quote_history = []  # position size in quote currency (e.g., SOL)
        cum_volume = 0.0          # cumulative volume counter
        tx_history = []
        strat_buy_times = []
        strat_buy_prices = []
        strat_sell_times = []
        strat_sell_prices = []
        strat_sell_reasons = []
        pending_orders = []
        last_price = None
        last_pool_base = None
        last_pool_quote = None
        price_impact = None

        deferred_sell_info = None
        deferred_trade_count = 0

        # Initialize tracking for target wallet
        target_wallet = "AGzrUzWwHFttUu446C31Pe3USoZMz8CB53mFp6upbhkA"
        target_pos = Position()
        target_tracking = False
        target_cum_volume = 0.0
        target_tx_history = []
        target_buy_times = []
        target_buy_prices = []
        target_sell_times = []
        target_sell_prices = []
        target_pnl_history = []
        target_position_history = []
        target_position_quote_history = []
        target_volume_history = []

        strat_cycle_sell_count = 0
        target_cycle_sell_count = 0

        for i, tx in feed_df.iterrows():
            current_slot = tx["slot"]
            # Execute pending orders due by this slot (delayed fills)
            pending_orders.sort(key=lambda o: o["target_slot"])
            while pending_orders and pending_orders[0]["target_slot"] <= current_slot:
                order = pending_orders.pop(0)

                side = order["side"].lower()
                if side == "buy":
                    fill_price = last_pool_quote / (last_pool_base - order["qty"])
                elif side == "sell":
                    fill_price = last_pool_quote / (last_pool_base + order["qty"])
                else:
                    fill_price = last_price

                if fill_price is None:
                    fill_price = ((tx["pool_quote"] - tx["amt_out"]) / (tx["pool_base"] + tx["amt_in"]) if tx["direction"] == "sell"
                                  else (tx["amt_in"] + tx["pool_quote"]) / (tx["pool_base"] - tx["amt_out"]))
                pos.apply_fill(side=order["side"], qty=order["qty"], price=fill_price)
                if order["side"].lower() == "buy":
                    strat_buy_times.append(tx["ts"])
                    strat_buy_prices.append(fill_price)
                elif order["side"].lower() == "sell":
                    strat_sell_times.append(tx["ts"])
                    strat_sell_prices.append(fill_price)
                    strat_sell_reasons.append(order.get("reason", ""))
                    strat_cycle_sell_count += 1
                    if pos.is_flat():
                        strat_cycle_sell_counts.append(strat_cycle_sell_count)
                        strat_cycle_sell_count = 0
                # Accumulate volume in quote currency for strategy orders
                if order["side"].lower() in ("buy", "sell"):
                    cum_volume += float(order["qty"]) * float(fill_price)
                    if feed_df.at[i, "trade_executed"] == "":
                        feed_df.at[i, "trade_executed"] = order["side"].lower()
                        feed_df.at[i, "trade_size"] = float(order["qty"]) * float(fill_price) / 1e9
                        feed_df.at[i, "trade_price"] = float(fill_price)
                        feed_df.at[i, "position"] = float(pos.qty) * float(fill_price) / 1e9
                    else:
                        feed_df.at[i, "trade_executed"] = feed_df.at[i, "trade_executed"] + "," + order["side"].lower()
                        feed_df.at[i, "trade_size"] = str(feed_df.at[i, "trade_size"]) + "," + str(float(order["qty"]) * float(fill_price) / 1e9)
                        feed_df.at[i, "trade_price"] = str(feed_df.at[i, "trade_price"]) + "," + str(float(fill_price))
                        feed_df.at[i, "position"] = str(feed_df.at[i, "position"]) + "," + str(float(pos.qty) * float(fill_price) / 1e9)

            if deferred_sell_info:
                if current_slot != deferred_sell_info["slot"] and deferred_trade_count < 2:
                    # Next slot arrived before two subsequent trades -> no sandwich, proceed with sell
                    target_slot = current_slot + s_delay
                    if not any(o["side"] == "sell" and o["target_slot"] == target_slot for o in pending_orders):
                        if deferred_sell_info["qty"] > float(pos.qty):
                            # Cap the sell order to the current position (skip if nothing left)
                            if float(pos.qty) <= 0:
                                # Nothing to sell – skip scheduling
                                deferred_sell_info = None
                            else:
                                deferred_sell_info["qty"] = float(pos.qty)
                        if deferred_sell_info is not None:
                            pending_orders.append({"side": "sell", "qty": deferred_sell_info["qty"], "target_slot": current_slot + s_delay, "reason": deferred_sell_info.get("reason", "")})
                            # Update strategy state as if a sell was executed
                            remaining_qty = float(pos.qty) - deferred_sell_info["qty"]
                            if remaining_qty > 0:
                                new_impact = (remaining_qty / deferred_sell_info["pool_base"]) if deferred_sell_info.get("pool_base") else float('inf')
                                state.num_remaining_order = max(1, math.ceil(new_impact / 0.003))
                            else:
                                state.num_remaining_order = 0
                            state.last_sell_price = deferred_sell_info["price"]
                            state.last_sell_ts = deferred_sell_info.get("ts", None)
                            state.highest_price_since_last_sell = deferred_sell_info["price"]
                            state.initial_threshold_active = False
                            if remaining_qty <= 0:
                                state.min_price = None
                                state.prior_avg_cost = None
                                state.num_remaining_order = None
                                state.last_sell_price = None
                                state.last_sell_ts = None
                                state.highest_price_since_last_sell = None
                                state.initial_threshold_active = True
                                state.buy_count = 0
                                state.skip_after_second = False
                    # If a sell order is already scheduled for this slot, skip adding the new sell
                    deferred_sell_info = None
                    deferred_trade_count = 0
                elif current_slot == deferred_sell_info["slot"]:
                    deferred_trade_count += 1
                    if deferred_trade_count == 2:
                        # Two subsequent trades in same slot -> check for sandwich attack
                        if tx["initiator"] == deferred_sell_info["initiator"] and abs(tx["amt_in"] - deferred_sell_info["amt_out"]) < 1e-9:
                            # Sandwich attack detected: cancel planned sell
                            # (Do not schedule the sell)
                            pass
                        else:
                            # Not a sandwich attack, proceed with sell
                            target_slot = current_slot + s_delay
                            if not any(o["side"] == "sell" and o["target_slot"] == target_slot for o in pending_orders):
                                if deferred_sell_info["qty"] > float(pos.qty):
                                    # Cap the sell order to the current position (skip if nothing left)
                                    if float(pos.qty) <= 0:
                                        # Nothing to sell – skip scheduling
                                        deferred_sell_info = None
                                    else:
                                        deferred_sell_info["qty"] = float(pos.qty)
                                if deferred_sell_info is not None:
                                    pending_orders.append({"side": "sell", "qty": deferred_sell_info["qty"], "target_slot": current_slot + s_delay, "reason": deferred_sell_info.get("reason", "")})
                                    # Update state for executed sell (same as above)
                                    remaining_qty = float(pos.qty) - deferred_sell_info["qty"]
                                    if remaining_qty > 0:
                                        new_impact = (remaining_qty / deferred_sell_info["pool_base"]) if deferred_sell_info.get("pool_base") else float('inf')
                                        state.num_remaining_order = max(1, math.ceil(new_impact / 0.003))
                                    else:
                                        state.num_remaining_order = 0
                                    state.last_sell_price = deferred_sell_info["price"]
                                    state.last_sell_ts = deferred_sell_info.get("ts", None)
                                    state.highest_price_since_last_sell = deferred_sell_info["price"]
                                    state.initial_threshold_active = False
                                    if remaining_qty <= 0:
                                        state.min_price = None
                                        state.prior_avg_cost = None
                                        state.num_remaining_order = None
                                        state.last_sell_price = None
                                        state.last_sell_ts = None
                                        state.highest_price_since_last_sell = None
                                        state.initial_threshold_active = True
                                        state.buy_count = 0
                                        state.skip_after_second = False
                        # If a sell order is already scheduled for this slot, skip adding the new sell
                        deferred_sell_info = None
                        deferred_trade_count = 0
            if tx["direction"] == "sell":
                # Large sell: check if we want to buy
                buy_decision = buy(tx, pos, state, feed_df, i)
                if buy_decision["buy"]:
                    # Apply the buy fill to position after a delay
                    target_slot = tx["slot"] + b_delay
                    if not any(o["side"] == "buy" and o["target_slot"] == target_slot for o in pending_orders):
                        pending_orders.append({"side": "buy", "qty": buy_decision["buy_qty"], "target_slot": target_slot})
                        feed_df.at[i, "trade_triggered"] = "buy"
                    # If a buy order is already scheduled for this slot, skip adding the new buy
                    strategy_impacts.append(buy_decision["impact"])
                # (If no buy, we simply skip)
            elif tx["direction"] == "buy":
                # Large buy: check if we want to sell
                sell_decision = sell(tx, pos, state)
                if sell_decision.get("defer_sell", False):
                    deferred_sell_info = {"qty": sell_decision["sell_qty"], "price": sell_decision["sell_price"], "initiator": tx["initiator"], "amt_out": tx["amt_out"], "slot": tx["slot"], "ts": tx.get("ts", None), "pool_base": tx["pool_base"], "reason": sell_decision.get("reason", "")}
                    deferred_trade_count = 0
                    feed_df.at[i, "trade_triggered"] = "sell"
                elif sell_decision["sell"]:
                    # (FIX) Adjust pending sells to avoid overselling
                    total_pending_sell_qty = sum(o["qty"] for o in pending_orders if o["side"] == "sell")
                    available_qty = float(pos.qty - Decimal(str(total_pending_sell_qty)))
                    if sell_decision["sell_qty"] > available_qty:
                        if available_qty <= 0:
                            sell_decision["sell"] = False
                            sell_decision["reason"] = sell_decision.get("reason", "") + " (skipped due to pending sells)"
                        else:
                            sell_decision["sell_qty"] = available_qty
                            # If this sell will close the entire position
                            if abs(available_qty - sell_decision["sell_qty"]) < 1e-9:
                                state.min_price = None
                                state.prior_avg_cost = None
                                state.num_remaining_order = None
                                state.last_sell_price = sell_decision["sell_price"]
                                state.last_sell_ts = tx.get("ts", None)
                                state.highest_price_since_last_sell = None
                                state.initial_threshold_active = True
                                state.buy_count = 0
                                state.skip_after_second = False
                            else:
                                remaining_qty = float(pos.qty) - sell_decision["sell_qty"]
                                new_impact = (remaining_qty / tx["pool_base"]) if tx["pool_base"] else float("inf")
                                state.num_remaining_order = max(1, math.ceil(new_impact / 0.003))
                                state.last_sell_price = sell_decision["sell_price"]
                                state.last_sell_ts = tx.get("ts", None)
                                state.highest_price_since_last_sell = sell_decision["sell_price"]
                                state.initial_threshold_active = False
                    # Schedule the sell (if not skipped)
                    if sell_decision["sell"]:
                        target_slot = tx["slot"] + s_delay
                        if not any(o["side"] == "sell" and o["target_slot"] == target_slot for o in pending_orders):
                            pending_orders.append({"side": "sell", "qty": sell_decision["sell_qty"], "target_slot": target_slot, "reason": sell_decision.get("reason", "")})
                            feed_df.at[i, "trade_triggered"] = "sell"

                        # If a sell order is already scheduled for this slot, skip adding the new sell
                # (If no sell, skip)
            # Record PnL and position after this swap
            tx_history.append(tx['ts'])
            pnl_history.append(float(pos.realised_pnl))
            position_history.append(float(pos.qty))
            volume_history.append(cum_volume)
            # Update last_price to the price after this swap (for use by delayed orders)
            if tx["direction"] == "sell":
                # Price after a sell swap (user sells base)
                last_price = ((tx["pool_quote"] - tx["amt_out"]) / (tx["pool_base"] + tx["amt_in"])) if tx["pool_base"] and tx["pool_quote"] else (tx["amt_out"] / tx["amt_in"] if tx["amt_in"] != 0 else 0.0)
                price_impact = -tx["amt_in"] / tx["pool_base"]
                last_pool_base = tx["pool_base"] + tx["amt_in"]
                last_pool_quote = tx["pool_quote"] - tx["amt_out"]

            elif tx["direction"] == "buy":
                # Price after a buy swap (user buys base)
                if (tx["pool_base"] - tx["amt_out"]) != 0:
                    last_price = (tx["amt_in"] + tx["pool_quote"]) / (tx["pool_base"] - tx["amt_out"])
                else:
                    last_price = float('inf')  # in case pool_base becomes zero (edge case)
                price_impact = tx["amt_in"] / tx["pool_quote"]
                last_pool_base = tx["pool_base"] - tx["amt_out"]
                last_pool_quote = tx["pool_quote"] + tx["amt_in"]
            # After updating price, track position in quote currency
            position_quote_history.append(float(pos.qty) * float(last_price))

            feed_df.at[i, "price"] = float(last_price)
            feed_df.at[i, "price_impact"] = float(price_impact)

            # Track target wallet's trades for analysis
            if tx.get("initiator") == target_wallet:
                if not target_tracking:
                    # Start tracking when target wallet's base reserve becomes 0
                    if tx["user_base_reserve"] == 0:
                        target_tracking = True
                        target_pos = Position()  # reset position at baseline
                        target_cum_volume = 0.0
                        # record baseline point with zero PnL and position
                        target_tx_history.append(tx["ts"])
                        target_pnl_history.append(0.0)
                        target_position_history.append(0.0)
                        target_position_quote_history.append(0.0)
                        target_volume_history.append(0.0)
                if target_tracking:
                    # Target wallet transaction after baseline
                    side = "buy" if tx["direction"] == "buy" else "sell"
                    if side == "buy":
                        qty = tx["amt_out"]  # base acquired by wallet
                        trade_price = (tx["amt_in"] / tx["amt_out"]) if tx["amt_out"] != 0 else 0.0
                        target_buy_times.append(tx["ts"])
                        target_buy_prices.append(float(trade_price))
                    else:
                        qty = tx["amt_in"]  # base sold by wallet
                        trade_price = (tx["amt_out"] / tx["amt_in"]) if tx["amt_in"] != 0 else 0.0
                        target_sell_times.append(tx["ts"])
                        target_sell_prices.append(float(trade_price))
                    if side == "sell" and qty > target_pos._qty:
                        qty = target_pos._qty
                    target_pos.apply_fill(side=side, qty=float(qty), price=float(trade_price))
                    if side == "sell":
                        target_cycle_sell_count += 1
                        if target_pos.is_flat():
                            target_cycle_sell_counts.append(target_cycle_sell_count)
                            target_cycle_sell_count = 0
                    target_cum_volume += float(qty) * float(trade_price)
                    target_tx_history.append(tx["ts"])
                    target_pnl_history.append(float(target_pos.realised_pnl))
                    target_position_history.append(float(target_pos.qty))
                    target_position_quote_history.append(float(target_pos.qty) * float(last_price))
                    target_volume_history.append(target_cum_volume)

        # After processing all swaps:
        # Execute any pending orders after processing all swaps
        while pending_orders:
            order = pending_orders.pop(0)
            # Fill at last known market price
            side = order["side"].lower()
            if side == "buy":
                fill_price = last_pool_quote / (last_pool_base - order["qty"])
            elif side == "sell":
                fill_price = last_pool_quote / (last_pool_base + order["qty"])
            else:
                fill_price = last_price
            pos.apply_fill(side=order["side"], qty=order["qty"], price=fill_price)
            if side == "sell":
                strat_cycle_sell_count += 1
                if pos.is_flat():
                    strat_cycle_sell_counts.append(strat_cycle_sell_count)
                    strat_cycle_sell_count = 0

        # **New Logic:** If position still open at end of data, close it
        if not pos.is_flat():  # if position still open at end of data
            # Sell all remaining position at current price
            final_qty = float(pos.qty)
            # Use last known market price for the sell
            final_price = last_price if last_price is not None else 0.0
            pos.apply_fill(side="sell", qty=final_qty, price=final_price)
            strat_cycle_sell_count += 1
            if pos.is_flat():
                strat_cycle_sell_counts.append(strat_cycle_sell_count)
                strat_cycle_sell_count = 0
            # Update cumulative volume for this final sell
            cum_volume += final_qty * float(final_price)
            last_row = feed_df.iloc[-1]
            strat_sell_times.append(last_row["ts"])
            strat_sell_prices.append(final_price)
            position_history.append(0)
            position_quote_history.append(0)
            volume_history.append(cum_volume)
            pnl_history.append(pos.realised_pnl)

        print(f"{file_name} - Final realized PnL: {pos.realised_pnl:.4f}")
        print(f"{file_name} - Final position quantity: {pos.qty}, average cost: {pos.avg_cost}")

        feed_df.to_csv(f"backtest_result-2/{file_name}_output_with_trades.csv", index=False)

        # Create subplots for price, PnL, and position
        # Create subplots for strategy and target wallet
        fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.01,
                             specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}],
                                    [{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]])

        # Prepare price high-low data by second
        price_data = []
        for _, tx in feed_df.iterrows():
            if tx["direction"] == "sell":
                price_after = ((tx["pool_quote"] - tx["amt_out"]) / (tx["pool_base"] + tx["amt_in"])) if tx["pool_base"] and tx["pool_quote"] else (tx["amt_out"] / tx["amt_in"] if tx["amt_in"] != 0 else 0.0)
            else:
                price_after = ((tx["amt_in"] + tx["pool_quote"]) / (tx["pool_base"] - tx["amt_out"])) if (tx["pool_base"] - tx["amt_out"]) != 0 else float('inf')
            price_data.append((tx["ts"], price_after))
        price_by_ts = {}
        for ts_val, price_val in price_data:
            if ts_val not in price_by_ts:
                price_by_ts[ts_val] = {"low": price_val, "high": price_val}
            else:
                price_by_ts[ts_val]["low"] = min(price_by_ts[ts_val]["low"], price_val)
                price_by_ts[ts_val]["high"] = max(price_by_ts[ts_val]["high"], price_val)
        price_x = []
        price_y = []
        for ts_val, vals in sorted(price_by_ts.items()):
            t_dt = pd.to_datetime(ts_val, unit='s')
            price_x += [t_dt, t_dt, None]
            price_y += [vals["low"], vals["high"], None]
        # Add price range trace (high-low bar) and strategy orders on subplot 1
        fig.add_trace(go.Scatter(x=price_x, y=price_y, mode='lines', name='Price Range', line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in strat_buy_times], y=strat_buy_prices,
                                 mode='markers', name='Buy Orders', marker=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in strat_sell_times], y=strat_sell_prices,
                                 mode='markers', name='Sell Orders', marker=dict(color='red'),
                                 text=strat_sell_reasons, hoverinfo='text'), row=1, col=1)
        fig.add_trace(go.Scatter(x=macd_x, y=macd_y, showlegend=True, mode="lines", line=dict(color="cyan"), 
                                 name="Macd Line"), row=1, col=1, secondary_y=True)
        # Add PnL trace on subplot 2
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in tx_history], y=position_history,
                                 mode='lines', name='Position'), row=2, col=1)
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in tx_history], y=position_quote_history,
                                 mode='lines', name='Position (quote)'), row=2, col=1, secondary_y=True)

        # Add position trace on subplot 3
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in tx_history], y=pnl_history,
                                 mode='lines', name='Cumulative PnL'), row=3, col=1)
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in tx_history], y=volume_history,
                                 mode='lines', name='Cumulative Volume'), row=3, col=1, secondary_y=True)
        
        # Add price range trace and target wallet orders on subplot 4
        fig.add_trace(go.Scatter(x=price_x, y=price_y, mode='lines', name='Price Range', line=dict(color='black'), showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in target_buy_times], y=target_buy_prices,
                                 mode='markers', name='Target Buy Orders', marker=dict(color='green')), row=4, col=1)
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in target_sell_times], y=target_sell_prices,
                                 mode='markers', name='Target Sell Orders', marker=dict(color='red')), row=4, col=1)
        # Add target wallet position traces on subplot 5
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in target_tx_history], y=target_position_history,
                                 mode='lines', line_shape='hv', name='Target Position'), row=5, col=1)
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in target_tx_history], y=target_position_quote_history,
                                 mode='lines', line_shape='hv', name='Target Position (quote)'), row=5, col=1, secondary_y=True)
        # Add target wallet PnL and volume traces on subplot 6
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in target_tx_history], y=target_pnl_history,
                                 mode='lines', name='Target Cumulative PnL'), row=6, col=1)
        fig.add_trace(go.Scatter(x=[pd.to_datetime(t, unit='s') for t in target_tx_history], y=target_volume_history,
                                 mode='lines', name='Target Cumulative Volume'), row=6, col=1, secondary_y=True)
        
        # Update axes titles
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Realized PnL (quote currency)', row=3, col=1)
        fig.update_yaxes(title_text='Cumulative Volume (quote currency)', row=3, col=1, secondary_y=True)
        fig.update_yaxes(title_text='Position (base units)', row=2, col=1)
        fig.update_yaxes(title_text='Position (quote currency)', row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text='Price', row=4, col=1)
        fig.update_yaxes(title_text='Position (base units)', row=5, col=1)
        fig.update_yaxes(title_text='Position (quote currency)', row=5, col=1, secondary_y=True)
        fig.update_yaxes(title_text='Realized PnL (quote currency)', row=6, col=1)
        fig.update_yaxes(title_text='Cumulative Volume (quote currency)', row=6, col=1, secondary_y=True)
        fig.update_xaxes(title_text='Time', row=6, col=1)

        fig.update_layout(
            height=2000)
        fig.write_html(f"backtest_result-2/{file_name}_plot.html")
        # fig.show()
        strategy_pnls.append(float(pos.realised_pnl))
        target_pnls.append(float(target_pos.realised_pnl))
        strategy_buys_counts.append(len(strat_buy_times))
        target_buys_counts.append(len(target_buy_times))

    # After processing all files, plot histogram of realized PnLs
    fig_hist = go.Figure()

    BIN = 1_000_000_000  # 1 B
    all_pnls = strategy_pnls + target_pnls          # combine the two lists/arrays
    left_edge = BIN * math.floor(min(all_pnls) / BIN)
    if left_edge == 0:
        left_edge -= BIN
    right_edge = BIN * math.ceil(max(all_pnls) / BIN)
    fig_hist.update_xaxes(range=[left_edge, right_edge])

    fig_hist.add_trace(go.Histogram(x=strategy_pnls, name="Strategy PnL", autobinx=False, xbins=dict(size=1e9)))
    fig_hist.add_trace(go.Histogram(x=target_pnls, name="Target Wallet PnL", autobinx=False, xbins=dict(size=1e9)))
    fig_hist.update_layout(barmode='group')
    total_strategy_pnl = sum(strategy_pnls)
    total_target_pnl = sum(target_pnls)
    pos_count_strategy = sum(1 for x in strategy_pnls if x > 0)
    neg_count_strategy = sum(1 for x in strategy_pnls if x < 0)
    zero_count_strategy = sum(1 for x in strategy_pnls if abs(x) < 1e-9)
    pos_count_target = sum(1 for x in target_pnls if x > 0)
    neg_count_target = sum(1 for x in target_pnls if x < 0)
    zero_count_target = sum(1 for x in target_pnls if abs(x) < 1e-9)
    higher_count = sum(1 for s, t in zip(strategy_pnls, target_pnls) if s > t)
    lower_count  = sum(1 for s, t in zip(strategy_pnls, target_pnls) if s < t)

    fig_hist.update_layout(
        title_text="Final Realized PnL Distribution",
        annotations=[
            dict(xref='paper', yref='paper', x=0.0, y=1.15, showarrow=False, align="left",
                 text=f"Strategy: total PnL={total_strategy_pnl:.2f}, positive PnL trades={pos_count_strategy}, negative PnL trades={neg_count_strategy}, zero PnL trades={zero_count_strategy}"),
            dict(xref='paper', yref='paper', x=0.0, y=1.08, showarrow=False, align="left",
                 text=f"Target: total PnL={total_target_pnl:.2f}, positive PnL trades={pos_count_target}, negative PnL trades={neg_count_target}, zero PnL trades={zero_count_target}"),
            dict(xref='paper', yref='paper', x=0.0, y=1.01, showarrow=False, align="left",
                 text=f"Strategy PnL higher than target in {higher_count} cases, lower in {lower_count} cases")
        ]
    )
    # Plot histogram for sell impact that triggers strategy buy orders
    fig_impacts = go.Figure()
    fig_impacts.add_trace(go.Histogram(x=[impact * 100 for impact in strategy_impacts],
                                       xbins=dict(start=1,size=1),
                                       name="Sell Impact (%)"))
    fig_impacts.update_layout(title_text="Sell Impact Triggering Strategy Buys",
                              xaxis_title="Sell Impact (%)",
                              yaxis_title="Frequency")
    # Plot histogram for number of buy orders per CSV
    fig_buys = go.Figure()
    fig_buys.add_trace(go.Histogram(x=strategy_buys_counts, xbins=dict(size=10),
                                    name="Strategy Buys per CSV"))
    fig_buys.add_trace(go.Histogram(x=target_buys_counts, xbins=dict(size=10),
                                    name="Target Wallet Buys per CSV"))
    fig_buys.update_layout(barmode='group',
                            title_text="Number of Buy Orders per CSV",
                            xaxis_title="Buy Orders per CSV",
                            yaxis_title="Frequency",
                            annotations=[
                                dict(xref='paper', yref='paper', x=0.0, y=1.15, showarrow=False, align="left",
                                     text=f"Strategy: total buy orders = {sum(strategy_buys_counts)}"),
                                dict(xref='paper', yref='paper', x=0.0, y=1.08, showarrow=False, align="left",
                                     text=f"Target: total buy orders = {sum(target_buys_counts)}")
                            ])
    # Plot histogram for number of sell orders per trading cycle
    fig_sells = go.Figure()
    fig_sells.add_trace(go.Histogram(x=strat_cycle_sell_counts, xbins=dict(size=1),
                                     name="Strategy Sells per Cycle"))
    fig_sells.add_trace(go.Histogram(x=target_cycle_sell_counts, xbins=dict(size=1),
                                     name="Target Sells per Cycle"))
    fig_sells.update_layout(barmode='group',
                             title_text="Number of Sell Orders per Trading Cycle",
                             xaxis_title="Sell Orders per Cycle",
                             yaxis_title="Frequency")
    fig_hist_html = fig_hist.to_html(full_html=True, include_plotlyjs=True)
    insert_index = fig_hist_html.rfind('</body>')
    combined_html = fig_hist_html[:insert_index] + \
                   fig_impacts.to_html(full_html=False, include_plotlyjs=False) + \
                   fig_buys.to_html(full_html=False, include_plotlyjs=False) + \
                   fig_sells.to_html(full_html=False, include_plotlyjs=False) + \
                   fig_hist_html[insert_index:]
    with open("backtest_result-2/pnl_summary.html", "w") as f:
        f.write(combined_html)
