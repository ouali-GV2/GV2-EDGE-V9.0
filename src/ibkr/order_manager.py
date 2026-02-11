"""
IBKR Order Manager - Advanced Order Management

Provides:
- Bracket orders (entry + stop + target)
- OCO (One-Cancels-Other) orders
- Trailing stops
- Order monitoring and alerts
- Position management
- Risk-based sizing
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Tuple, Any
import asyncio
import logging

from .connector import (
    IBKRConnector,
    Order,
    OrderType,
    OrderAction,
    OrderStatus,
    Position
)

logger = logging.getLogger(__name__)


class BracketType(Enum):
    """Bracket order types."""
    FIXED = "FIXED"           # Fixed stop and target
    PERCENT = "PERCENT"       # Percentage-based
    ATR = "ATR"               # ATR-based stops
    TRAILING = "TRAILING"     # Trailing stop


class PositionSizeMethod(Enum):
    """Position sizing methods."""
    FIXED_SHARES = "FIXED_SHARES"
    FIXED_DOLLAR = "FIXED_DOLLAR"
    PERCENT_EQUITY = "PERCENT_EQUITY"
    RISK_BASED = "RISK_BASED"  # Based on stop distance


@dataclass
class BracketOrder:
    """Bracket order (entry + stop + target)."""
    id: str
    symbol: str
    action: OrderAction
    quantity: int

    # Entry
    entry_type: OrderType
    entry_price: Optional[float] = None

    # Stop loss
    stop_price: float = 0.0
    stop_type: OrderType = OrderType.STOP

    # Take profit
    target_price: Optional[float] = None
    target_type: OrderType = OrderType.LIMIT

    # Status
    status: str = "PENDING"  # PENDING, ACTIVE, FILLED, CANCELLED
    entry_filled: bool = False
    stop_filled: bool = False
    target_filled: bool = False

    # Order IDs
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    target_order_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

    # PnL
    entry_fill_price: float = 0.0
    exit_fill_price: float = 0.0
    realized_pnl: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "action": self.action.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "status": self.status,
            "realized_pnl": self.realized_pnl,
        }


@dataclass
class RiskParameters:
    """Risk management parameters."""
    # Position limits
    max_position_size: int = 1000
    max_position_value: float = 10000.0
    max_positions: int = 5

    # Risk per trade
    max_risk_per_trade: float = 100.0  # Max $ to risk
    max_risk_percent: float = 1.0       # Max % of equity to risk

    # Stop loss
    default_stop_percent: float = 2.0
    max_stop_percent: float = 5.0

    # Daily limits
    max_daily_loss: float = 500.0
    max_daily_trades: int = 20


@dataclass
class OrderManagerConfig:
    """Configuration for order manager."""
    # Risk parameters
    risk: RiskParameters = field(default_factory=RiskParameters)

    # Order defaults
    default_tif: str = "DAY"
    use_adaptive_orders: bool = False

    # Monitoring
    monitor_interval: float = 1.0  # seconds
    auto_manage_brackets: bool = True


class IBKROrderManager:
    """
    Advanced order management for IBKR.

    Usage:
        connector = get_ibkr_connector()
        await connector.connect()

        manager = IBKROrderManager(connector)

        # Place bracket order
        bracket = await manager.place_bracket(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            entry_price=150.0,
            stop_price=147.0,
            target_price=156.0
        )

        # Risk-based sizing
        shares = manager.calculate_position_size(
            symbol="AAPL",
            entry_price=150.0,
            stop_price=147.0,
            method=PositionSizeMethod.RISK_BASED
        )

        # Close position
        await manager.close_position("AAPL")
    """

    def __init__(
        self,
        connector: IBKRConnector,
        config: Optional[OrderManagerConfig] = None
    ):
        self.config = config or OrderManagerConfig()
        self._connector = connector

        # Bracket orders
        self._brackets: Dict[str, BracketOrder] = {}
        self._bracket_counter = 0

        # Daily tracking
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._daily_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0)

        # Callbacks
        self._fill_callbacks: List[Callable] = []
        self._pnl_callbacks: List[Callable] = []

        # Monitoring
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    # Position Sizing

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        method: PositionSizeMethod = PositionSizeMethod.RISK_BASED,
        fixed_shares: int = 100,
        fixed_dollars: float = 5000.0,
        percent_equity: float = 5.0
    ) -> int:
        """
        Calculate position size based on method.

        Args:
            symbol: Stock symbol
            entry_price: Planned entry price
            stop_price: Stop loss price
            method: Sizing method
            fixed_shares: For FIXED_SHARES method
            fixed_dollars: For FIXED_DOLLAR method
            percent_equity: For PERCENT_EQUITY method

        Returns:
            Number of shares to trade
        """
        risk = self.config.risk
        account = self._connector.get_account_summary()

        if method == PositionSizeMethod.FIXED_SHARES:
            shares = fixed_shares

        elif method == PositionSizeMethod.FIXED_DOLLAR:
            shares = int(fixed_dollars / entry_price)

        elif method == PositionSizeMethod.PERCENT_EQUITY:
            equity = account.net_liquidation
            position_value = equity * (percent_equity / 100)
            shares = int(position_value / entry_price)

        elif method == PositionSizeMethod.RISK_BASED:
            # Calculate shares based on risk per trade
            risk_per_share = abs(entry_price - stop_price)
            if risk_per_share <= 0:
                logger.warning("Invalid stop distance for risk-based sizing")
                return 0

            # Use smaller of fixed risk or percent risk
            max_risk = min(
                risk.max_risk_per_trade,
                account.net_liquidation * (risk.max_risk_percent / 100)
            )

            shares = int(max_risk / risk_per_share)

        else:
            shares = fixed_shares

        # Apply limits
        shares = min(shares, risk.max_position_size)
        shares = min(shares, int(risk.max_position_value / entry_price))

        # Check buying power
        required = shares * entry_price
        if required > account.buying_power:
            shares = int(account.buying_power / entry_price)

        return max(0, shares)

    # Bracket Orders

    async def place_bracket(
        self,
        symbol: str,
        action: OrderAction,
        quantity: int,
        entry_price: Optional[float] = None,
        stop_price: float = 0.0,
        target_price: Optional[float] = None,
        entry_type: OrderType = OrderType.LIMIT,
        bracket_type: BracketType = BracketType.FIXED
    ) -> Optional[BracketOrder]:
        """
        Place a bracket order (entry + stop + target).

        Args:
            symbol: Stock symbol
            action: BUY or SELL
            quantity: Number of shares
            entry_price: Entry price (None for market order)
            stop_price: Stop loss price
            target_price: Take profit price (optional)
            entry_type: Entry order type
            bracket_type: Type of bracket

        Returns:
            BracketOrder if placed successfully
        """
        # Validate
        if not self._connector.is_connected():
            logger.error("Not connected to IBKR")
            return None

        if not self._check_daily_limits():
            logger.error("Daily limits exceeded")
            return None

        # Create bracket
        self._bracket_counter += 1
        bracket_id = f"bracket_{self._bracket_counter}"

        bracket = BracketOrder(
            id=bracket_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            entry_type=entry_type,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price
        )

        try:
            # Place entry order
            entry_order = await self._connector.place_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=entry_type,
                limit_price=entry_price
            )

            if not entry_order:
                logger.error("Failed to place entry order")
                return None

            bracket.entry_order_id = entry_order.id
            bracket.status = "ACTIVE"

            # Store bracket
            self._brackets[bracket_id] = bracket

            # Place stop and target will be done after entry fills
            # (if auto_manage_brackets is True)
            if self.config.auto_manage_brackets:
                self._connector.subscribe_orders(
                    lambda o: self._on_order_update(o, bracket_id)
                )

            logger.info(
                f"Placed bracket order: {bracket_id} - "
                f"{action.value} {quantity} {symbol} @ {entry_price}"
            )

            self._daily_trades += 1
            return bracket

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            return None

    async def _place_exit_orders(self, bracket: BracketOrder) -> None:
        """Place stop and target orders after entry fills."""
        # Determine exit action
        exit_action = OrderAction.SELL if bracket.action == OrderAction.BUY else OrderAction.BUY

        # Place stop loss
        stop_order = await self._connector.place_order(
            symbol=bracket.symbol,
            action=exit_action,
            quantity=bracket.quantity,
            order_type=bracket.stop_type,
            stop_price=bracket.stop_price
        )

        if stop_order:
            bracket.stop_order_id = stop_order.id

        # Place target if specified
        if bracket.target_price:
            target_order = await self._connector.place_order(
                symbol=bracket.symbol,
                action=exit_action,
                quantity=bracket.quantity,
                order_type=bracket.target_type,
                limit_price=bracket.target_price
            )

            if target_order:
                bracket.target_order_id = target_order.id

        logger.info(f"Exit orders placed for bracket {bracket.id}")

    def _on_order_update(self, order: Order, bracket_id: str) -> None:
        """Handle order update for bracket."""
        bracket = self._brackets.get(bracket_id)
        if not bracket:
            return

        # Entry filled
        if order.id == bracket.entry_order_id and order.status == OrderStatus.FILLED:
            bracket.entry_filled = True
            bracket.entry_fill_price = order.avg_fill_price
            bracket.filled_at = datetime.now()

            # Place exit orders
            asyncio.create_task(self._place_exit_orders(bracket))

        # Stop filled
        elif order.id == bracket.stop_order_id and order.status == OrderStatus.FILLED:
            bracket.stop_filled = True
            bracket.exit_fill_price = order.avg_fill_price
            bracket.status = "STOPPED"

            # Cancel target
            if bracket.target_order_id:
                asyncio.create_task(
                    self._connector.cancel_order(bracket.target_order_id)
                )

            self._calculate_bracket_pnl(bracket)

        # Target filled
        elif order.id == bracket.target_order_id and order.status == OrderStatus.FILLED:
            bracket.target_filled = True
            bracket.exit_fill_price = order.avg_fill_price
            bracket.status = "TARGET_HIT"

            # Cancel stop
            if bracket.stop_order_id:
                asyncio.create_task(
                    self._connector.cancel_order(bracket.stop_order_id)
                )

            self._calculate_bracket_pnl(bracket)

    def _calculate_bracket_pnl(self, bracket: BracketOrder) -> None:
        """Calculate PnL for completed bracket."""
        if bracket.entry_fill_price and bracket.exit_fill_price:
            if bracket.action == OrderAction.BUY:
                pnl = (bracket.exit_fill_price - bracket.entry_fill_price) * bracket.quantity
            else:
                pnl = (bracket.entry_fill_price - bracket.exit_fill_price) * bracket.quantity

            bracket.realized_pnl = pnl
            self._daily_pnl += pnl

            logger.info(f"Bracket {bracket.id} closed with PnL: ${pnl:.2f}")

            # Notify callbacks
            for callback in self._pnl_callbacks:
                try:
                    callback(bracket)
                except Exception as e:
                    logger.error(f"PnL callback error: {e}")

    # Position Management

    async def close_position(
        self,
        symbol: str,
        order_type: OrderType = OrderType.MARKET
    ) -> Optional[Order]:
        """Close entire position for a symbol."""
        position = self._connector.get_position(symbol)
        if not position or position.quantity == 0:
            logger.warning(f"No position to close for {symbol}")
            return None

        # Determine action
        action = OrderAction.SELL if position.quantity > 0 else OrderAction.BUY
        quantity = abs(position.quantity)

        order = await self._connector.place_order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type=order_type
        )

        if order:
            logger.info(f"Closing position: {action.value} {quantity} {symbol}")

        return order

    async def close_all_positions(self) -> List[Order]:
        """Close all positions."""
        orders = []
        positions = self._connector.get_positions()

        for symbol, position in positions.items():
            if position.quantity != 0:
                order = await self.close_position(symbol)
                if order:
                    orders.append(order)

        return orders

    async def flatten(self) -> Tuple[List[Order], int]:
        """
        Flatten all positions and cancel all open orders.

        Returns:
            Tuple of (orders placed, orders cancelled)
        """
        # Cancel all open orders
        cancelled = 0
        for order in self._connector.get_open_orders():
            if await self._connector.cancel_order(order.id):
                cancelled += 1

        # Close all positions
        orders = await self.close_all_positions()

        # Cancel bracket exit orders
        for bracket in self._brackets.values():
            if bracket.status == "ACTIVE":
                if bracket.stop_order_id:
                    await self._connector.cancel_order(bracket.stop_order_id)
                if bracket.target_order_id:
                    await self._connector.cancel_order(bracket.target_order_id)
                bracket.status = "CANCELLED"

        logger.info(f"Flattened: {len(orders)} positions closed, {cancelled} orders cancelled")
        return orders, cancelled

    # Risk Management

    def _check_daily_limits(self) -> bool:
        """Check if daily limits allow trading."""
        # Reset if new day
        now = datetime.now()
        if now.date() > self._daily_reset.date():
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_reset = now.replace(hour=0, minute=0, second=0)

        # Check loss limit
        if self._daily_pnl <= -self.config.risk.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return False

        # Check trade limit
        if self._daily_trades >= self.config.risk.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return False

        return True

    def get_daily_stats(self) -> Dict:
        """Get daily trading statistics."""
        return {
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "max_daily_loss": self.config.risk.max_daily_loss,
            "max_daily_trades": self.config.risk.max_daily_trades,
            "loss_limit_remaining": self.config.risk.max_daily_loss + self._daily_pnl,
            "trades_remaining": self.config.risk.max_daily_trades - self._daily_trades,
        }

    # Bracket management

    def get_bracket(self, bracket_id: str) -> Optional[BracketOrder]:
        """Get bracket order by ID."""
        return self._brackets.get(bracket_id)

    def get_active_brackets(self) -> List[BracketOrder]:
        """Get all active bracket orders."""
        return [b for b in self._brackets.values() if b.status == "ACTIVE"]

    async def cancel_bracket(self, bracket_id: str) -> bool:
        """Cancel a bracket order."""
        bracket = self._brackets.get(bracket_id)
        if not bracket:
            return False

        # Cancel all component orders
        if bracket.entry_order_id:
            await self._connector.cancel_order(bracket.entry_order_id)
        if bracket.stop_order_id:
            await self._connector.cancel_order(bracket.stop_order_id)
        if bracket.target_order_id:
            await self._connector.cancel_order(bracket.target_order_id)

        bracket.status = "CANCELLED"
        return True

    async def modify_stop(
        self,
        bracket_id: str,
        new_stop_price: float
    ) -> bool:
        """Modify stop price for a bracket."""
        bracket = self._brackets.get(bracket_id)
        if not bracket or not bracket.stop_order_id:
            return False

        # Cancel old stop
        await self._connector.cancel_order(bracket.stop_order_id)

        # Place new stop
        exit_action = OrderAction.SELL if bracket.action == OrderAction.BUY else OrderAction.BUY
        stop_order = await self._connector.place_order(
            symbol=bracket.symbol,
            action=exit_action,
            quantity=bracket.quantity,
            order_type=bracket.stop_type,
            stop_price=new_stop_price
        )

        if stop_order:
            bracket.stop_order_id = stop_order.id
            bracket.stop_price = new_stop_price
            logger.info(f"Modified stop for {bracket_id} to {new_stop_price}")
            return True

        return False

    # Callbacks

    def on_fill(self, callback: Callable[[Order], None]) -> None:
        """Subscribe to fill events."""
        self._fill_callbacks.append(callback)

    def on_pnl(self, callback: Callable[[BracketOrder], None]) -> None:
        """Subscribe to PnL events."""
        self._pnl_callbacks.append(callback)

    # Stats

    def get_stats(self) -> Dict:
        """Get order manager statistics."""
        active_brackets = self.get_active_brackets()
        return {
            "total_brackets": len(self._brackets),
            "active_brackets": len(active_brackets),
            **self.get_daily_stats()
        }


# Factory function
def create_order_manager(
    connector: IBKRConnector,
    config: Optional[OrderManagerConfig] = None
) -> IBKROrderManager:
    """Create an order manager instance."""
    return IBKROrderManager(connector, config)
