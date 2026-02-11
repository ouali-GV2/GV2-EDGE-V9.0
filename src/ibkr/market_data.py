"""
IBKR Market Data - Real-time and Historical Market Data

Provides:
- Real-time quotes with Level 1 data
- Historical bars (OHLCV)
- Market scanners
- Options chains
- Fundamental data
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import asyncio
import logging

logger = logging.getLogger(__name__)


class BarSize(Enum):
    """Bar size for historical data."""
    SEC_1 = "1 secs"
    SEC_5 = "5 secs"
    SEC_10 = "10 secs"
    SEC_15 = "15 secs"
    SEC_30 = "30 secs"
    MIN_1 = "1 min"
    MIN_2 = "2 mins"
    MIN_3 = "3 mins"
    MIN_5 = "5 mins"
    MIN_10 = "10 mins"
    MIN_15 = "15 mins"
    MIN_20 = "20 mins"
    MIN_30 = "30 mins"
    HOUR_1 = "1 hour"
    HOUR_2 = "2 hours"
    HOUR_3 = "3 hours"
    HOUR_4 = "4 hours"
    HOUR_8 = "8 hours"
    DAY_1 = "1 day"
    WEEK_1 = "1 week"
    MONTH_1 = "1 month"


class WhatToShow(Enum):
    """Data type for historical data."""
    TRADES = "TRADES"
    MIDPOINT = "MIDPOINT"
    BID = "BID"
    ASK = "ASK"
    BID_ASK = "BID_ASK"
    ADJUSTED_LAST = "ADJUSTED_LAST"
    HISTORICAL_VOLATILITY = "HISTORICAL_VOLATILITY"
    OPTION_IMPLIED_VOLATILITY = "OPTION_IMPLIED_VOLATILITY"


@dataclass
class Bar:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    wap: float = 0.0  # Weighted average price
    bar_count: int = 0

    @property
    def range(self) -> float:
        """Bar range (high - low)."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Candle body size."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Check if bullish candle."""
        return self.close > self.open

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class TickData:
    """Real-time tick data."""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Prices
    bid: float = 0.0
    bid_size: int = 0
    ask: float = 0.0
    ask_size: int = 0
    last: float = 0.0
    last_size: int = 0

    # Daily stats
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0  # Previous close
    volume: int = 0

    # Calculated
    mid: float = 0.0
    spread: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0

    # Halted flag
    halted: bool = False

    def update_calculated(self) -> None:
        """Update calculated fields."""
        if self.bid and self.ask:
            self.mid = (self.bid + self.ask) / 2
            self.spread = self.ask - self.bid
        if self.close > 0 and self.last > 0:
            self.change = self.last - self.close
            self.change_pct = (self.change / self.close) * 100


@dataclass
class ScannerResult:
    """Market scanner result."""
    rank: int
    symbol: str
    distance: float = 0.0  # Distance from scan criteria

    # Price data
    last: float = 0.0
    change_pct: float = 0.0
    volume: int = 0

    # Additional fields depending on scan type
    extra: Dict[str, Any] = field(default_factory=dict)


class MarketScanner(Enum):
    """Built-in market scanners."""
    TOP_GAINERS = "TOP_PERC_GAIN"
    TOP_LOSERS = "TOP_PERC_LOSE"
    MOST_ACTIVE = "MOST_ACTIVE"
    HOT_BY_VOLUME = "HOT_BY_VOLUME"
    HOT_BY_PRICE = "HOT_BY_PRICE"
    TOP_VOLUME_RATE = "TOP_VOLUME_RATE"
    HIGH_VOLATILITY = "HIGH_VS_13W_HL"
    LOW_VOLATILITY = "LOW_VS_13W_HL"
    HALTED = "HALTED"
    HIGH_DIVIDEND = "HIGH_DIVIDEND_YIELD"
    HIGH_PE = "HIGH_PE_RATIO"
    LOW_PE = "LOW_PE_RATIO"


@dataclass
class MarketDataConfig:
    """Configuration for market data."""
    # Subscription settings
    market_data_type: int = 1  # 1=Live, 3=Delayed
    generic_tick_list: str = ""  # Additional tick types

    # Historical data
    use_rth: bool = True  # Regular trading hours only

    # Scanner
    scanner_rows: int = 50


class IBKRMarketData:
    """
    IBKR market data manager.

    Usage:
        connector = get_ibkr_connector()
        await connector.connect()

        market_data = IBKRMarketData(connector)

        # Real-time quotes
        def on_tick(tick: TickData):
            print(f"{tick.symbol}: {tick.last}")

        await market_data.subscribe("AAPL", on_tick)

        # Historical data
        bars = await market_data.get_historical(
            "AAPL",
            duration="1 D",
            bar_size=BarSize.MIN_5
        )

        # Scanner
        gainers = await market_data.scan(MarketScanner.TOP_GAINERS)
    """

    def __init__(
        self,
        connector,  # IBKRConnector
        config: Optional[MarketDataConfig] = None
    ):
        self.config = config or MarketDataConfig()
        self._connector = connector

        # Subscriptions
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._tick_data: Dict[str, TickData] = {}

        # Contracts cache
        self._contracts: Dict[str, Any] = {}

    async def _get_contract(self, symbol: str):
        """Get or create qualified contract."""
        if symbol in self._contracts:
            return self._contracts[symbol]

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            await self._connector._ib.qualifyContractsAsync(contract)
            self._contracts[symbol] = contract
            return contract

        except Exception as e:
            logger.error(f"Failed to qualify contract for {symbol}: {e}")
            return None

    # Real-time data

    async def subscribe(
        self,
        symbol: str,
        callback: Callable[[TickData], None]
    ) -> bool:
        """
        Subscribe to real-time tick data.

        Args:
            symbol: Stock symbol
            callback: Function to call on tick updates

        Returns:
            True if subscribed successfully
        """
        if not self._connector.is_connected():
            logger.error("Not connected to IBKR")
            return False

        try:
            contract = await self._get_contract(symbol)
            if not contract:
                return False

            ib = self._connector._ib

            # Request market data
            ticker = ib.reqMktData(
                contract,
                genericTickList=self.config.generic_tick_list
            )

            # Initialize tick data
            self._tick_data[symbol] = TickData(symbol=symbol)

            # Setup handler
            def on_pending_tickers(tickers):
                for t in tickers:
                    if t.contract.symbol == symbol:
                        tick = self._tick_data.get(symbol)
                        if tick:
                            tick.timestamp = datetime.now()
                            tick.bid = t.bid or tick.bid
                            tick.bid_size = t.bidSize or tick.bid_size
                            tick.ask = t.ask or tick.ask
                            tick.ask_size = t.askSize or tick.ask_size
                            tick.last = t.last or tick.last
                            tick.last_size = t.lastSize or tick.last_size
                            tick.open = t.open or tick.open
                            tick.high = t.high or tick.high
                            tick.low = t.low or tick.low
                            tick.close = t.close or tick.close
                            tick.volume = int(t.volume or tick.volume)
                            tick.halted = getattr(t, 'halted', 0) > 0
                            tick.update_calculated()

                            callback(tick)

            ib.pendingTickersEvent += on_pending_tickers

            # Store subscription
            if symbol not in self._subscriptions:
                self._subscriptions[symbol] = []
            self._subscriptions[symbol].append(callback)

            logger.info(f"Subscribed to market data for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False

    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from market data."""
        if symbol in self._subscriptions:
            del self._subscriptions[symbol]
        if symbol in self._tick_data:
            del self._tick_data[symbol]
        # Note: Would need to cancel market data request in IB

    def get_tick(self, symbol: str) -> Optional[TickData]:
        """Get current tick data for symbol."""
        return self._tick_data.get(symbol)

    # Historical data

    async def get_historical(
        self,
        symbol: str,
        duration: str = "1 D",
        bar_size: BarSize = BarSize.MIN_5,
        what_to_show: WhatToShow = WhatToShow.TRADES,
        end_date: Optional[datetime] = None,
        use_rth: Optional[bool] = None
    ) -> List[Bar]:
        """
        Get historical bar data.

        Args:
            symbol: Stock symbol
            duration: Duration string (e.g., "1 D", "1 W", "1 M", "1 Y")
            bar_size: Bar size enum
            what_to_show: Data type
            end_date: End date (default: now)
            use_rth: Use regular trading hours only

        Returns:
            List of Bar objects
        """
        if not self._connector.is_connected():
            logger.error("Not connected to IBKR")
            return []

        try:
            contract = await self._get_contract(symbol)
            if not contract:
                return []

            ib = self._connector._ib
            end_date = end_date or datetime.now()
            use_rth = use_rth if use_rth is not None else self.config.use_rth

            # Request historical data
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size.value,
                whatToShow=what_to_show.value,
                useRTH=use_rth,
                formatDate=1
            )

            # Convert to Bar objects
            result = []
            for bar in bars:
                result.append(Bar(
                    timestamp=bar.date,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=int(bar.volume),
                    wap=bar.average,
                    bar_count=bar.barCount
                ))

            logger.debug(f"Got {len(result)} bars for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []

    async def get_daily_bars(
        self,
        symbol: str,
        days: int = 30
    ) -> List[Bar]:
        """Get daily bars for N days."""
        return await self.get_historical(
            symbol,
            duration=f"{days} D",
            bar_size=BarSize.DAY_1
        )

    async def get_intraday_bars(
        self,
        symbol: str,
        bar_size: BarSize = BarSize.MIN_5
    ) -> List[Bar]:
        """Get today's intraday bars."""
        return await self.get_historical(
            symbol,
            duration="1 D",
            bar_size=bar_size,
            use_rth=True
        )

    # Market scanners

    async def scan(
        self,
        scanner_type: MarketScanner,
        above_price: float = 1.0,
        below_price: float = 500.0,
        above_volume: int = 100000,
        market_cap_above: float = 0,
        market_cap_below: float = 0,
        limit: Optional[int] = None
    ) -> List[ScannerResult]:
        """
        Run a market scanner.

        Args:
            scanner_type: Type of scanner
            above_price: Minimum price filter
            below_price: Maximum price filter
            above_volume: Minimum volume filter
            market_cap_above: Minimum market cap
            market_cap_below: Maximum market cap
            limit: Max results

        Returns:
            List of ScannerResult objects
        """
        if not self._connector.is_connected():
            logger.error("Not connected to IBKR")
            return []

        try:
            from ib_insync import ScannerSubscription

            ib = self._connector._ib

            # Create scanner subscription
            scan = ScannerSubscription(
                instrument="STK",
                locationCode="STK.US.MAJOR",
                scanCode=scanner_type.value,
                abovePrice=above_price,
                belowPrice=below_price,
                aboveVolume=above_volume,
                marketCapAbove=market_cap_above if market_cap_above > 0 else "",
                marketCapBelow=market_cap_below if market_cap_below > 0 else "",
                numberOfRows=limit or self.config.scanner_rows
            )

            # Run scanner
            results = await ib.reqScannerDataAsync(scan)

            # Convert to ScannerResult
            scanner_results = []
            for i, item in enumerate(results):
                scanner_results.append(ScannerResult(
                    rank=i + 1,
                    symbol=item.contractDetails.contract.symbol,
                    distance=item.distance,
                    extra={
                        "con_id": item.contractDetails.contract.conId,
                        "exchange": item.contractDetails.contract.exchange,
                    }
                ))

            logger.info(f"Scanner {scanner_type.value} returned {len(scanner_results)} results")
            return scanner_results

        except Exception as e:
            logger.error(f"Scanner failed: {e}")
            return []

    async def get_top_gainers(self, limit: int = 20) -> List[ScannerResult]:
        """Get top percentage gainers."""
        return await self.scan(MarketScanner.TOP_GAINERS, limit=limit)

    async def get_top_losers(self, limit: int = 20) -> List[ScannerResult]:
        """Get top percentage losers."""
        return await self.scan(MarketScanner.TOP_LOSERS, limit=limit)

    async def get_most_active(self, limit: int = 20) -> List[ScannerResult]:
        """Get most active by volume."""
        return await self.scan(MarketScanner.MOST_ACTIVE, limit=limit)

    async def get_halted(self) -> List[ScannerResult]:
        """Get currently halted stocks."""
        return await self.scan(MarketScanner.HALTED)

    # Utility

    def is_halted(self, symbol: str) -> bool:
        """Check if symbol is halted."""
        tick = self._tick_data.get(symbol)
        return tick.halted if tick else False

    def get_stats(self) -> Dict:
        """Get market data statistics."""
        return {
            "subscriptions": len(self._subscriptions),
            "contracts_cached": len(self._contracts),
            "ticks_tracked": len(self._tick_data),
        }


# Factory function
def create_market_data(connector, config: Optional[MarketDataConfig] = None) -> IBKRMarketData:
    """Create a market data instance."""
    return IBKRMarketData(connector, config)
