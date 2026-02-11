"""
IBKR Connector - Interactive Brokers TWS/Gateway Connection

Handles connection to Interactive Brokers:
- TWS (Trader Workstation) or IB Gateway connection
- Real-time market data subscription
- News feed subscription
- Order placement and management
- Account/Portfolio data

Requires: ib_insync library (pip install ib_insync)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Set
import asyncio
import logging

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state to TWS/Gateway."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class OrderType(Enum):
    """Order types."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    TRAIL = "TRAIL"
    TRAIL_LIMIT = "TRAIL LIMIT"


class OrderAction(Enum):
    """Order actions."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SSHORT"


@dataclass
class ConnectionConfig:
    """Configuration for IBKR connection."""
    host: str = "127.0.0.1"
    port: int = 7497          # TWS paper: 7497, TWS live: 7496, Gateway: 4001/4002
    client_id: int = 1
    timeout: int = 30
    readonly: bool = False    # True = no order placement

    # Auto-reconnect
    auto_reconnect: bool = True
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10

    # Market data
    market_data_type: int = 1  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen


@dataclass
class Quote:
    """Real-time quote data."""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Prices
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0

    # Volume
    volume: int = 0
    avg_volume: int = 0

    # Calculated
    spread: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0

    def update(self) -> None:
        """Update calculated fields."""
        self.spread = self.ask - self.bid if self.ask and self.bid else 0
        if self.close > 0:
            self.change = self.last - self.close
            self.change_pct = (self.change / self.close) * 100


@dataclass
class Position:
    """Account position."""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def current_price(self) -> float:
        """Estimate current price from market value."""
        if self.quantity != 0:
            return self.market_value / self.quantity
        return 0.0


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType

    # Prices
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None

    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # IBKR specific
    ibkr_order_id: Optional[int] = None
    ibkr_perm_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "action": self.action.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
        }


@dataclass
class AccountSummary:
    """Account summary data."""
    account_id: str = ""
    net_liquidation: float = 0.0
    total_cash: float = 0.0
    buying_power: float = 0.0
    gross_position_value: float = 0.0
    maintenance_margin: float = 0.0
    available_funds: float = 0.0
    excess_liquidity: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    last_updated: datetime = field(default_factory=datetime.now)


class IBKRConnector:
    """
    Interactive Brokers TWS/Gateway connector.

    Usage:
        connector = IBKRConnector()
        await connector.connect()

        # Subscribe to quotes
        connector.subscribe_quote("AAPL", callback=on_quote)

        # Place order
        order = await connector.place_order(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.00
        )

        # Get positions
        positions = connector.get_positions()
    """

    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig()

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._ib = None  # ib_insync.IB instance
        self._connected_at: Optional[datetime] = None
        self._reconnect_count = 0

        # Subscriptions
        self._quote_subscriptions: Dict[str, List[Callable]] = {}
        self._news_subscriptions: List[Callable] = []
        self._order_subscriptions: List[Callable] = []

        # Cache
        self._quotes: Dict[str, Quote] = {}
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._account: AccountSummary = AccountSummary()

        # Order ID counter
        self._order_counter = 0

    async def connect(self) -> bool:
        """
        Connect to TWS/Gateway.

        Returns:
            True if connected successfully
        """
        if self._state == ConnectionState.CONNECTED:
            return True

        self._state = ConnectionState.CONNECTING

        try:
            # Import ib_insync
            try:
                from ib_insync import IB
            except ImportError:
                logger.error("ib_insync not installed. Run: pip install ib_insync")
                self._state = ConnectionState.ERROR
                return False

            self._ib = IB()

            # Connect
            await asyncio.wait_for(
                self._ib.connectAsync(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=self.config.client_id,
                    readonly=self.config.readonly
                ),
                timeout=self.config.timeout
            )

            # Set market data type
            self._ib.reqMarketDataType(self.config.market_data_type)

            # Setup event handlers
            self._setup_handlers()

            self._state = ConnectionState.CONNECTED
            self._connected_at = datetime.now()
            self._reconnect_count = 0

            logger.info(f"Connected to IBKR at {self.config.host}:{self.config.port}")

            # Initial data fetch
            await self._fetch_account_data()
            await self._fetch_positions()

            return True

        except asyncio.TimeoutError:
            logger.error(f"Connection timeout after {self.config.timeout}s")
            self._state = ConnectionState.ERROR
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._state = ConnectionState.ERROR

            if self.config.auto_reconnect:
                await self._schedule_reconnect()

            return False

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

        self._state = ConnectionState.DISCONNECTED
        self._connected_at = None
        logger.info("Disconnected from IBKR")

    def _setup_handlers(self) -> None:
        """Setup IB event handlers."""
        if not self._ib:
            return

        # Error handler
        self._ib.errorEvent += self._on_error

        # Order updates
        self._ib.orderStatusEvent += self._on_order_status

        # Position updates
        self._ib.positionEvent += self._on_position

        # Account updates
        self._ib.accountValueEvent += self._on_account_value

        # Disconnect handler
        self._ib.disconnectedEvent += self._on_disconnect

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """Handle IB errors."""
        # Common non-critical errors
        non_critical = {2104, 2106, 2158}  # Market data farm messages

        if errorCode in non_critical:
            logger.debug(f"IBKR info: {errorCode} - {errorString}")
        else:
            logger.error(f"IBKR error {errorCode}: {errorString}")

    def _on_order_status(self, trade) -> None:
        """Handle order status updates."""
        order_id = str(trade.order.orderId)

        if order_id in self._orders:
            order = self._orders[order_id]

            # Update status
            status_map = {
                "Submitted": OrderStatus.SUBMITTED,
                "Filled": OrderStatus.FILLED,
                "Cancelled": OrderStatus.CANCELLED,
                "PendingSubmit": OrderStatus.PENDING,
                "PreSubmitted": OrderStatus.PENDING,
            }
            order.status = status_map.get(trade.orderStatus.status, OrderStatus.PENDING)
            order.filled_quantity = int(trade.orderStatus.filled)
            order.avg_fill_price = trade.orderStatus.avgFillPrice

            if order.status == OrderStatus.FILLED:
                order.filled_at = datetime.now()

            # Notify subscribers
            for callback in self._order_subscriptions:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Order callback error: {e}")

    def _on_position(self, position) -> None:
        """Handle position updates."""
        symbol = position.contract.symbol

        pos = Position(
            symbol=symbol,
            quantity=int(position.position),
            avg_cost=position.avgCost,
            market_value=position.position * position.avgCost
        )

        self._positions[symbol] = pos

    def _on_account_value(self, value) -> None:
        """Handle account value updates."""
        key = value.tag
        val = value.value

        try:
            val = float(val)
        except ValueError:
            return

        mapping = {
            "NetLiquidation": "net_liquidation",
            "TotalCashValue": "total_cash",
            "BuyingPower": "buying_power",
            "GrossPositionValue": "gross_position_value",
            "MaintMarginReq": "maintenance_margin",
            "AvailableFunds": "available_funds",
            "ExcessLiquidity": "excess_liquidity",
            "UnrealizedPnL": "unrealized_pnl",
            "RealizedPnL": "realized_pnl",
        }

        if key in mapping:
            setattr(self._account, mapping[key], val)
            self._account.last_updated = datetime.now()

    async def _on_disconnect(self) -> None:
        """Handle disconnection."""
        logger.warning("Disconnected from IBKR")
        self._state = ConnectionState.DISCONNECTED

        if self.config.auto_reconnect:
            await self._schedule_reconnect()

    async def _schedule_reconnect(self) -> None:
        """Schedule reconnection attempt."""
        if self._reconnect_count >= self.config.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return

        self._reconnect_count += 1
        delay = self.config.reconnect_delay * self._reconnect_count

        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_count})")
        await asyncio.sleep(delay)
        await self.connect()

    async def _fetch_account_data(self) -> None:
        """Fetch initial account data."""
        if not self._ib or not self._ib.isConnected():
            return

        try:
            accounts = self._ib.managedAccounts()
            if accounts:
                self._account.account_id = accounts[0]
                self._ib.reqAccountUpdates(True, self._account.account_id)
        except Exception as e:
            logger.error(f"Failed to fetch account data: {e}")

    async def _fetch_positions(self) -> None:
        """Fetch current positions."""
        if not self._ib or not self._ib.isConnected():
            return

        try:
            positions = self._ib.positions()
            for pos in positions:
                self._on_position(pos)
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")

    # Market Data

    async def subscribe_quote(
        self,
        symbol: str,
        callback: Callable[[Quote], None]
    ) -> bool:
        """
        Subscribe to real-time quotes for a symbol.

        Args:
            symbol: Stock symbol
            callback: Function to call on quote updates

        Returns:
            True if subscribed successfully
        """
        if not self._ib or not self._ib.isConnected():
            logger.error("Not connected to IBKR")
            return False

        try:
            from ib_insync import Stock

            # Create contract
            contract = Stock(symbol, "SMART", "USD")

            # Qualify contract
            await self._ib.qualifyContractsAsync(contract)

            # Request market data
            ticker = self._ib.reqMktData(contract)

            # Setup quote handler
            def on_tick(ticker):
                quote = Quote(
                    symbol=symbol,
                    bid=ticker.bid or 0,
                    ask=ticker.ask or 0,
                    last=ticker.last or 0,
                    close=ticker.close or 0,
                    open=ticker.open or 0,
                    high=ticker.high or 0,
                    low=ticker.low or 0,
                    volume=int(ticker.volume or 0),
                )
                quote.update()

                self._quotes[symbol] = quote
                callback(quote)

            ticker.updateEvent += on_tick

            # Store subscription
            if symbol not in self._quote_subscriptions:
                self._quote_subscriptions[symbol] = []
            self._quote_subscriptions[symbol].append(callback)

            logger.info(f"Subscribed to quotes for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False

    def unsubscribe_quote(self, symbol: str) -> None:
        """Unsubscribe from quotes."""
        if symbol in self._quote_subscriptions:
            del self._quote_subscriptions[symbol]
            # Note: Would need to cancel market data request

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get cached quote for symbol."""
        return self._quotes.get(symbol)

    # Orders

    async def place_order(
        self,
        symbol: str,
        action: OrderAction,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_amount: Optional[float] = None,
        tif: str = "DAY"
    ) -> Optional[Order]:
        """
        Place an order.

        Args:
            symbol: Stock symbol
            action: BUY, SELL, or SHORT
            quantity: Number of shares
            order_type: Order type (MARKET, LIMIT, etc.)
            limit_price: Limit price for LIMIT orders
            stop_price: Stop price for STOP orders
            trail_amount: Trail amount for TRAIL orders
            tif: Time in force (DAY, GTC, IOC, etc.)

        Returns:
            Order object if placed successfully
        """
        if self.config.readonly:
            logger.error("Cannot place orders in readonly mode")
            return None

        if not self._ib or not self._ib.isConnected():
            logger.error("Not connected to IBKR")
            return None

        try:
            from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder, StopLimitOrder

            # Create contract
            contract = Stock(symbol, "SMART", "USD")
            await self._ib.qualifyContractsAsync(contract)

            # Create IB order
            action_str = action.value
            if order_type == OrderType.MARKET:
                ib_order = MarketOrder(action_str, quantity)
            elif order_type == OrderType.LIMIT:
                ib_order = LimitOrder(action_str, quantity, limit_price)
            elif order_type == OrderType.STOP:
                ib_order = StopOrder(action_str, quantity, stop_price)
            elif order_type == OrderType.STOP_LIMIT:
                ib_order = StopLimitOrder(action_str, quantity, limit_price, stop_price)
            else:
                ib_order = MarketOrder(action_str, quantity)

            ib_order.tif = tif

            # Place order
            trade = self._ib.placeOrder(contract, ib_order)

            # Create our order object
            self._order_counter += 1
            order = Order(
                id=f"order_{self._order_counter}",
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                trail_amount=trail_amount,
                status=OrderStatus.SUBMITTED,
                submitted_at=datetime.now(),
                ibkr_order_id=trade.order.orderId
            )

            self._orders[order.id] = order

            logger.info(f"Placed order: {order.id} - {action.value} {quantity} {symbol}")

            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if not order.ibkr_order_id:
            return False

        try:
            # Find the trade
            for trade in self._ib.openTrades():
                if trade.order.orderId == order.ibkr_order_id:
                    self._ib.cancelOrder(trade.order)
                    order.status = OrderStatus.CANCELLED
                    return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")

        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        return [
            o for o in self._orders.values()
            if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
        ]

    # Positions and Account

    def get_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self._positions.copy()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def get_account_summary(self) -> AccountSummary:
        """Get account summary."""
        return self._account

    def get_buying_power(self) -> float:
        """Get current buying power."""
        return self._account.buying_power

    def get_cash(self) -> float:
        """Get available cash."""
        return self._account.total_cash

    # Order subscriptions

    def subscribe_orders(self, callback: Callable[[Order], None]) -> None:
        """Subscribe to order updates."""
        self._order_subscriptions.append(callback)

    # Status

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    def get_state(self) -> ConnectionState:
        """Get connection state."""
        return self._state

    def get_stats(self) -> Dict:
        """Get connector statistics."""
        return {
            "state": self._state.value,
            "connected_at": self._connected_at.isoformat() if self._connected_at else None,
            "quote_subscriptions": len(self._quote_subscriptions),
            "positions": len(self._positions),
            "open_orders": len(self.get_open_orders()),
            "account_id": self._account.account_id,
        }


# Singleton instance
_connector: Optional[IBKRConnector] = None


def get_ibkr_connector(config: Optional[ConnectionConfig] = None) -> IBKRConnector:
    """Get singleton IBKRConnector instance."""
    global _connector
    if _connector is None:
        _connector = IBKRConnector(config)
    return _connector
