"""Open, close, and monitor MT5 positions."""
import logging
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5

log = logging.getLogger(__name__)

MAGIC = 200001


class PositionManager:
    def __init__(self, market, config: dict, state=None) -> None:
        self.market = market
        self.config = config
        self.state = state
        self.magic = config.get("magic_number", MAGIC)

    def open_position(
        self,
        direction: str,
        lot: float,
        sl: float,
        tp: float,
        comment: str = "v2",
        dry_run: bool = False,
    ) -> Optional[int]:
        """Open a BUY or SELL position. Returns ticket number or None."""
        tick_data = self.market.get_tick()
        if tick_data is None:
            log.error("Cannot open position: no tick data")
            return None
        bid, ask, _ = tick_data

        if direction == "buy":
            price = ask
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = bid
            order_type = mt5.ORDER_TYPE_SELL

        if dry_run:
            log.info("[DRY-RUN] Would open %s @ %.2f  SL=%.2f  TP=%.2f  lot=%.2f",
                     direction.upper(), price, sl, tp, lot)
            return -1

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.market.symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.market.get_filling_mode(),
        }

        result = mt5.order_send(request)
        if result is None:
            log.error("order_send returned None for %s", direction)
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("order_send failed: retcode=%s comment=%s", result.retcode, result.comment)
            return None

        ticket = result.order
        trade_info = {
            "direction": direction,
            "entry_price": price,
            "open_time": datetime.now(timezone.utc).isoformat(),
            "lot": lot,
            "sl": sl,
            "tp": tp,
        }

        # Persist to state for restart recovery
        if self.state:
            self.state.track_position(ticket, trade_info)

        log.info("Opened %s ticket=%d price=%.2f sl=%.2f tp=%.2f lot=%.2f",
                 direction.upper(), ticket, price, sl, tp, lot)
        return ticket

    def close_position(self, ticket: int, dry_run: bool = False) -> Optional[float]:
        """Close a position by ticket. Returns profit or None."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            log.warning("close_position: ticket %d not found", ticket)
            if self.state:
                self.state.untrack_position(ticket)
            return None

        pos = positions[0]
        tick_data = self.market.get_tick()
        if tick_data is None:
            log.error("Cannot close position: no tick data")
            return None
        bid, ask, _ = tick_data

        if pos.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = ask

        if dry_run:
            log.info("[DRY-RUN] Would close ticket %d @ %.2f", ticket, price)
            if self.state:
                self.state.untrack_position(ticket)
            return 0.0

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.market.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic,
            "comment": "v2-close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.market.get_filling_mode(),
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            log.error("close_position failed: ticket=%d retcode=%s", ticket, retcode)
            return None

        if self.state:
            self.state.untrack_position(ticket)
        log.info("Closed ticket=%d price=%.2f", ticket, price)
        return pos.profit

    def check_closed_positions(self, market) -> list:
        """Detect positions closed by SL/TP. Returns enriched close dicts."""
        closed = []
        if not self.state:
            return closed

        tracked = self.state.get_tracked_positions()
        if not tracked:
            return closed

        open_tickets = {p.ticket for p in market.get_positions()}

        for ticket_str, info in list(tracked.items()):
            ticket = int(ticket_str)
            if ticket not in open_tickets:
                # Position is gone — closed by SL/TP/manual
                close_data = self._build_close_data(ticket, info, market)
                self.state.untrack_position(ticket)
                closed.append(close_data)

        return closed

    def _build_close_data(self, ticket: int, info: dict, market) -> dict:
        """Build enriched close event data with PnL and duration."""
        import time as time_module
        now = datetime.now(timezone.utc)
        entry_price = info.get("entry_price", 0)
        direction = info.get("direction", "unknown")
        lot = info.get("lot", 0)
        open_time_str = info.get("open_time", "")

        # Calculate duration
        duration_seconds = 0
        if open_time_str:
            try:
                open_time = datetime.fromisoformat(open_time_str)
                duration_seconds = int((now - open_time).total_seconds())
            except (ValueError, TypeError):
                pass

        # Try to get close details from MT5 deal history with retry
        close_price = 0.0
        profit = 0.0
        swap = 0.0
        close_reason = "sl_or_tp"
        
        # Retry up to 3 times with small delays to let deal propagate
        for attempt in range(3):
            try:
                from datetime import timedelta
                deals = mt5.history_deals_get(
                    now - timedelta(days=7), now,
                    position=ticket
                )
                if deals:
                    for deal in reversed(deals):
                        # Verify this deal actually belongs to our position
                        if deal.entry == 1 and deal.position_id == ticket:  # DEAL_ENTRY_OUT
                            close_price = deal.price
                            profit = deal.profit
                            swap = deal.swap
                            if deal.reason == 3:  # DEAL_REASON_SL
                                close_reason = "stop_loss"
                            elif deal.reason == 4:  # DEAL_REASON_TP
                                close_reason = "take_profit"
                            elif deal.reason == 0:  # DEAL_REASON_CLIENT
                                close_reason = "manual"
                            break
                    
                    if close_price != 0.0:
                        break  # Found valid data, stop retrying
                
                if attempt < 2:
                    time_module.sleep(0.5)  # Brief delay before retry
                    
            except Exception as e:
                log.warning("Attempt %d: Could not fetch deal history for ticket %d: %s", attempt + 1, ticket, e)
        
        # Fallback: if deal not found, estimate from SL/TP
        if close_price == 0.0:
            log.warning("Deal history not found for ticket %d after retries, using balance-based estimation", ticket)
            acct = market.get_account()
            balance_after = acct[0] if acct else 0
            # We can infer profit from balance change, but we don't have the previous balance reliably
            # Use SL/TP as best guess for close price
            sl = info.get("sl", 0)
            tp = info.get("tp", 0)
            
            # Check balance change to determine if SL or TP hit
            # This is a rough heuristic
            if balance_after > 0 and entry_price > 0:
                # Try to get the actual deal without position filter (broader search)
                try:
                    from datetime import timedelta
                    recent_deals = mt5.history_deals_get(
                        now - timedelta(minutes=5), now
                    )
                    if recent_deals:
                        for deal in reversed(recent_deals):
                            if (deal.entry == 1 and 
                                deal.symbol == self.market.symbol and
                                abs(deal.volume - lot) < 0.01):
                                close_price = deal.price
                                profit = deal.profit
                                swap = deal.swap
                                if deal.reason == 3:
                                    close_reason = "stop_loss"
                                elif deal.reason == 4:
                                    close_reason = "take_profit"
                                elif deal.reason == 0:
                                    close_reason = "manual"
                                log.info("Found deal via broad search for ticket %d: price=%.2f profit=%.2f", 
                                        ticket, close_price, profit)
                                break
                except Exception as e:
                    log.warning("Broad deal search failed for ticket %d: %s", ticket, e)

        # Calculate pips
        profit_pips = 0.0
        if close_price and entry_price:
            if direction == "buy":
                profit_pips = (close_price - entry_price) / 1.0
            else:
                profit_pips = (entry_price - close_price) / 1.0

        # Get balance after close
        acct = market.get_account()
        balance_after = acct[0] if acct else 0

        return {
            "ticket": ticket,
            "direction": direction,
            "entry_price": entry_price,
            "close_price": close_price,
            "profit": round(profit, 2),
            "profit_pips": round(profit_pips, 2),
            "lot": lot,
            "sl": info.get("sl", 0),
            "tp": info.get("tp", 0),
            "swap": round(swap, 2),
            "duration_seconds": duration_seconds,
            "close_reason": close_reason,
            "balance_after": balance_after,
        }

    def recover_positions(self, market) -> int:
        """On startup, scan MT5 for open positions with our magic and track them."""
        if not self.state:
            return 0

        tracked = self.state.get_tracked_positions()
        open_positions = market.get_positions()
        recovered = 0

        for pos in open_positions:
            ticket_str = str(pos.ticket)
            if ticket_str not in tracked:
                info = {
                    "direction": "buy" if pos.type == 0 else "sell",
                    "entry_price": pos.price_open,
                    "open_time": datetime.fromtimestamp(pos.time, tz=timezone.utc).isoformat(),
                    "lot": pos.volume,
                    "sl": pos.sl,
                    "tp": pos.tp,
                }
                self.state.track_position(pos.ticket, info)
                recovered += 1
                log.info("Recovered position: ticket=%d %s @ %.2f",
                         pos.ticket, info["direction"].upper(), pos.price_open)

        return recovered

    def calculate_sl_tp(self, direction: str, price: float, atr: float) -> tuple:
        """Return (sl_price, tp_price) based on ATR multipliers."""
        sl_dist = atr * self.config.get("sl_atr_multiplier", 2.0)
        tp_dist = atr * self.config.get("tp_atr_multiplier", 3.0)
        if direction == "buy":
            return price - sl_dist, price + tp_dist
        else:
            return price + sl_dist, price - tp_dist
