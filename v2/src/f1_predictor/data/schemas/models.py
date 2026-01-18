"""Pydantic models for F1 data structures."""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class CircuitLocation(BaseModel):
    """Circuit location information."""

    lat: float
    long: float
    locality: str
    country: str


class Circuit(BaseModel):
    """Circuit information."""

    circuit_id: str = Field(alias="circuitId")
    circuit_name: str = Field(alias="circuitName")
    url: Optional[str] = None
    location: Optional[CircuitLocation] = Field(default=None, alias="Location")

    # Additional scraped fields
    circuit_type: Optional[str] = None  # "Street" or "Road"
    direction: Optional[str] = None  # "Clockwise" or "Counter-clockwise"
    length: Optional[float] = None  # km

    class Config:
        populate_by_name = True


class Driver(BaseModel):
    """Driver information."""

    driver_id: str = Field(alias="driverId")
    permanent_number: Optional[str] = Field(default=None, alias="permanentNumber")
    code: Optional[str] = None
    url: Optional[str] = None
    given_name: str = Field(alias="givenName")
    family_name: str = Field(alias="familyName")
    date_of_birth: date = Field(alias="dateOfBirth")
    nationality: str

    class Config:
        populate_by_name = True

    @property
    def full_name(self) -> str:
        """Get driver's full name."""
        return f"{self.given_name} {self.family_name}"

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def parse_date(cls, v: str | date) -> date:
        if isinstance(v, date):
            return v
        return datetime.strptime(v, "%Y-%m-%d").date()


class Constructor(BaseModel):
    """Constructor/team information."""

    constructor_id: str = Field(alias="constructorId")
    url: Optional[str] = None
    name: str
    nationality: str

    class Config:
        populate_by_name = True


class Race(BaseModel):
    """Race event information."""

    season: int
    round: int
    url: Optional[str] = None
    race_name: str = Field(alias="raceName")
    circuit: Circuit = Field(alias="Circuit")
    date: date
    time: Optional[str] = None

    class Config:
        populate_by_name = True

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | date) -> date:
        if isinstance(v, date):
            return v
        return datetime.strptime(v, "%Y-%m-%d").date()

    @field_validator("season", "round", mode="before")
    @classmethod
    def parse_int(cls, v: str | int) -> int:
        return int(v)


class RaceResult(BaseModel):
    """Individual race result for a driver."""

    number: Optional[str] = None
    position: Optional[int] = None
    position_text: str = Field(alias="positionText")
    points: float
    driver: Driver = Field(alias="Driver")
    constructor: Constructor = Field(alias="Constructor")
    grid: int
    laps: int
    status: str
    time_millis: Optional[int] = Field(default=None, alias="Time")
    fastest_lap: Optional[dict] = Field(default=None, alias="FastestLap")

    class Config:
        populate_by_name = True

    @field_validator("grid", "laps", mode="before")
    @classmethod
    def parse_int(cls, v: str | int) -> int:
        return int(v)

    @field_validator("points", mode="before")
    @classmethod
    def parse_float(cls, v: str | float) -> float:
        return float(v)

    @field_validator("position", mode="before")
    @classmethod
    def parse_position(cls, v: str | int | None) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    @field_validator("time_millis", mode="before")
    @classmethod
    def parse_time(cls, v: dict | int | None) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, dict):
            return v.get("millis")
        return v


class QualifyingResult(BaseModel):
    """Qualifying session result for a driver."""

    number: Optional[str] = None
    position: int
    driver: Driver = Field(alias="Driver")
    constructor: Constructor = Field(alias="Constructor")
    q1: Optional[str] = Field(default=None, alias="Q1")
    q2: Optional[str] = Field(default=None, alias="Q2")
    q3: Optional[str] = Field(default=None, alias="Q3")

    class Config:
        populate_by_name = True

    @field_validator("position", mode="before")
    @classmethod
    def parse_int(cls, v: str | int) -> int:
        return int(v)


class Standing(BaseModel):
    """Championship standing entry."""

    position: int
    position_text: str = Field(alias="positionText")
    points: float
    wins: int
    driver: Optional[Driver] = Field(default=None, alias="Driver")
    constructor: Optional[Constructor] = Field(default=None, alias="Constructor")
    constructors: Optional[list[Constructor]] = Field(default=None, alias="Constructors")

    class Config:
        populate_by_name = True

    @field_validator("position", "wins", mode="before")
    @classmethod
    def parse_int(cls, v: str | int) -> int:
        return int(v)

    @field_validator("points", mode="before")
    @classmethod
    def parse_float(cls, v: str | float) -> float:
        return float(v)


class Weather(BaseModel):
    """Weather information for a race."""

    season: int
    round: int
    race_name: str
    weather_text: Optional[str] = None
    source: str = "wikipedia"  # or "f1fansite"

    class Config:
        populate_by_name = True
