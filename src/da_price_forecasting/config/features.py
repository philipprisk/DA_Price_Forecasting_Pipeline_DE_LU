from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CovariateName = Literal[
    "exaa",
    "load_forecast",
    "ntc",
    "generation_unavailability",
    "renewable_generation_proxy",
    "commodities",
    "foreign_day_ahead_prices",
    "foreign_day_ahead_price_spreads",
]


class CommodityInstrumentConfig(BaseModel):
    name: str
    column: str | None = None
    ticker: str | None = None
    investing_id: int | None = None
    price_field: str | None = None


class CommodityConfig(BaseModel):
    provider: Literal["yfinance", "investiny"] = "yfinance"
    lag_days: int = 2
    lookback_days: int = 14
    auto_adjust: bool = False
    progress: bool = False
    request_timeout_seconds: float = 30.0
    instruments: list[CommodityInstrumentConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_instruments(self) -> "CommodityConfig":
        for instrument in self.instruments:
            if self.provider == "yfinance" and not instrument.ticker:
                raise ValueError("yfinance commodity instruments require `ticker`.")
            if self.provider == "investiny" and instrument.investing_id is None:
                raise ValueError("investiny commodity instruments require `investing_id`.")
        return self


class CovariateConfig(BaseModel):
    """User-facing covariate source selection shared by model families."""

    covariates: list[CovariateName] = Field(default_factory=list)
    ntc_neighbors: list[str] = Field(
        default_factory=lambda: ["FR", "NL", "AT", "CH", "PL", "CZ", "DK_1", "DK_2"]
    )
    renewable_proxy_file: str = "data/processed/renewable_proxy/dwd_icon_c5_renewable_proxy.csv"
    foreign_price_markets: list[str] = Field(default_factory=list)
    commodities: CommodityConfig = Field(default_factory=CommodityConfig)

    @field_validator("covariates", mode="before")
    @classmethod
    def _coerce_covariates(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    @field_validator("ntc_neighbors", mode="before")
    @classmethod
    def _coerce_ntc_neighbors(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    @field_validator("foreign_price_markets", mode="before")
    @classmethod
    def _coerce_foreign_price_markets(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)
