"""
Combined Design Patterns for AI Agents
======================================
This script combines three design patterns:
1. Strategy Pattern - Swappable agent personalities
2. Chain of Responsibility - Sequential handlers with shared context
3. Observer Pattern - Logging and monitoring agent calls
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

load_dotenv()


# ==================================
# SHARED DEPENDENCIES
# ==================================


@dataclass
class TravelDeps:
    user_name: str
    origin_city: str


# ==================================
# SHARED CONTEXT (Chain of Responsibility)
# ==================================


class TripContext(BaseModel):
    """Accumulated context passed through the chain of handlers."""

    destination: Optional[str] = None
    destination_message: Optional[str] = None
    from_city: Optional[str] = None
    arrival_time: Optional[str] = None
    hotel_name: Optional[str] = None
    hotel_location: Optional[str] = None
    hotel_price: Optional[int] = None
    activities: list[str] = Field(default_factory=list)


# ==================================
# OUTPUT SCHEMAS
# ==================================


class DestinationOutput(BaseModel):
    destination: str = Field(..., description="The recommended destination")
    message: str = Field(..., description="Personalized message about the destination")


class FlightOutput(BaseModel):
    from_city: str
    to_city: str
    arrival_time: str
    airline: str


class HotelOutput(BaseModel):
    name: str
    location: str
    price_per_night_usd: int
    stars: int


class ActivitiesOutput(BaseModel):
    activities: list[str] = Field(..., description="List of recommended activities")


# ==================================
# OBSERVER PATTERN
# ==================================


class AgentCallObserver(Protocol):
    """Protocol for observing agent calls."""

    def notify(
        self,
        agent_name: str,
        prompt: str,
        deps: TravelDeps,
        output: BaseModel,
        duration: float,
    ) -> None: ...


class ConsoleLogger(AgentCallObserver):
    """Logs agent calls to console."""

    def notify(
        self,
        agent_name: str,
        prompt: str,
        deps: TravelDeps,
        output: BaseModel,
        duration: float,
    ) -> None:
        print(f"\n  📋 [LOG] {agent_name} completed in {duration:.2f}s")


class DetailedLogger(AgentCallObserver):
    """Logs detailed information about agent calls."""

    def notify(
        self,
        agent_name: str,
        prompt: str,
        deps: TravelDeps,
        output: BaseModel,
        duration: float,
    ) -> None:
        print("\n" + "=" * 50)
        print(f"📋 AGENT CALL LOG")
        print("=" * 50)
        print(f"Agent: {agent_name}")
        print(f"User: {deps.user_name} | Origin: {deps.origin_city}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"Output: {output.model_dump()}")
        print(f"Duration: {duration:.2f}s")
        print("=" * 50)


class MetricsCollector(AgentCallObserver):
    """Collects metrics about agent performance."""

    def __init__(self):
        self.calls: list[dict] = []

    def notify(
        self,
        agent_name: str,
        prompt: str,
        deps: TravelDeps,
        output: BaseModel,
        duration: float,
    ) -> None:
        self.calls.append(
            {
                "agent": agent_name,
                "duration": duration,
                "user": deps.user_name,
            }
        )

    def summary(self) -> None:
        if not self.calls:
            print("No metrics collected.")
            return
        total_time = sum(c["duration"] for c in self.calls)
        print("\n📊 METRICS SUMMARY")
        print(f"  Total calls: {len(self.calls)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per call: {total_time / len(self.calls):.2f}s")
        for call in self.calls:
            print(f"    - {call['agent']}: {call['duration']:.2f}s")


# ==================================
# STRATEGY PATTERN - Agent Personalities
# ==================================


class AgentPersonality(Enum):
    PROFESSIONAL = "professional"
    FUN = "fun"
    BUDGET = "budget"


def get_personality_prompt(personality: AgentPersonality) -> str:
    """Returns the personality modifier for system prompts."""
    prompts = {
        AgentPersonality.PROFESSIONAL: (
            "You are highly professional and polite. "
            "Give thoughtful, detailed recommendations."
        ),
        AgentPersonality.FUN: (
            "You are fun and quirky! Get super excited about recommendations. "
            "Use casual language and occasional humor."
        ),
        AgentPersonality.BUDGET: (
            "You are a frugal expert focused on value and affordability. "
            "Always highlight cost-saving options."
        ),
    }
    return prompts[personality]


# Strategy factories - create agents with different personalities
def create_destination_agent(
    personality: AgentPersonality,
) -> Agent[TravelDeps, DestinationOutput]:
    base_prompt = "You help users select an ideal travel destination based on their preferences."
    return Agent(
        "openai:gpt-4o",
        name=f"DestinationAgent-{personality.value}",
        deps_type=TravelDeps,
        output_type=DestinationOutput,
        system_prompt=f"{base_prompt} {get_personality_prompt(personality)}",
    )


def create_flight_agent(
    personality: AgentPersonality,
) -> Agent[TravelDeps, FlightOutput]:
    base_prompt = "Plan realistic flight itineraries for trips."
    return Agent(
        "openai:gpt-4o",
        name=f"FlightAgent-{personality.value}",
        deps_type=TravelDeps,
        output_type=FlightOutput,
        system_prompt=f"{base_prompt} {get_personality_prompt(personality)}",
    )


def create_hotel_agent(
    personality: AgentPersonality,
) -> Agent[TravelDeps, HotelOutput]:
    base_prompt = "Suggest good hotels near airports or city centers."
    return Agent(
        "openai:gpt-4o",
        name=f"HotelAgent-{personality.value}",
        deps_type=TravelDeps,
        output_type=HotelOutput,
        system_prompt=f"{base_prompt} {get_personality_prompt(personality)}",
    )


def create_activity_agent(
    personality: AgentPersonality,
) -> Agent[TravelDeps, ActivitiesOutput]:
    base_prompt = "Suggest local activities suitable for travelers."
    return Agent(
        "openai:gpt-4o",
        name=f"ActivityAgent-{personality.value}",
        deps_type=TravelDeps,
        output_type=ActivitiesOutput,
        system_prompt=f"{base_prompt} {get_personality_prompt(personality)}",
    )


# ==================================
# OBSERVED AGENT RUNNER
# ==================================


async def run_agent_with_observers(
    agent: Agent,
    prompt: str,
    deps: TravelDeps,
    observers: list[AgentCallObserver],
) -> BaseModel:
    """Runs an agent and notifies all observers."""
    start = time.perf_counter()
    result = await agent.run(prompt, deps=deps)
    duration = time.perf_counter() - start

    for observer in observers:
        observer.notify(
            agent_name=agent.name or "UnnamedAgent",
            prompt=prompt,
            deps=deps,
            output=result.output,
            duration=duration,
        )

    return result.output


# ==================================
# CHAIN OF RESPONSIBILITY - Handlers
# ==================================

# Type alias for handler functions
# Handler = Callable[
#     [str, TravelDeps, TripContext, AgentPersonality, list[AgentCallObserver]],
#     None,
# ]


async def handle_destination(
    user_input: str,
    deps: TravelDeps,
    ctx: TripContext,
    personality: AgentPersonality,
    observers: list[AgentCallObserver],
) -> None:
    """Step 1: Choose destination."""
    agent = create_destination_agent(personality)
    output: DestinationOutput = await run_agent_with_observers(
        agent, user_input, deps, observers
    )
    ctx.destination = output.destination
    ctx.destination_message = output.message
    print(f"📍 Destination: {ctx.destination}")
    print(f"   Message: {ctx.destination_message}")


async def handle_flight(
    user_input: str,
    deps: TravelDeps,
    ctx: TripContext,
    personality: AgentPersonality,
    observers: list[AgentCallObserver],
) -> None:
    """Step 2: Plan flight using context from step 1."""
    agent = create_flight_agent(personality)
    prompt = f"Plan a flight from {deps.origin_city} to {ctx.destination}."
    output: FlightOutput = await run_agent_with_observers(agent, prompt, deps, observers)
    ctx.from_city = output.from_city
    ctx.arrival_time = output.arrival_time
    print(f"✈️  Flight: {output.airline} from {ctx.from_city} → {ctx.destination}")
    print(f"   Arriving at: {ctx.arrival_time}")


async def handle_hotel(
    user_input: str,
    deps: TravelDeps,
    ctx: TripContext,
    personality: AgentPersonality,
    observers: list[AgentCallObserver],
) -> None:
    """Step 3: Book hotel using context from steps 1-2."""
    agent = create_hotel_agent(personality)
    prompt = (
        f"Recommend a hotel in {ctx.destination} for a traveler "
        f"arriving at {ctx.arrival_time}."
    )
    output: HotelOutput = await run_agent_with_observers(agent, prompt, deps, observers)
    ctx.hotel_name = output.name
    ctx.hotel_location = output.location
    ctx.hotel_price = output.price_per_night_usd
    print(f"🏨 Hotel: {ctx.hotel_name} ({output.stars}★)")
    print(f"   Location: {ctx.hotel_location} | ${ctx.hotel_price}/night")


async def handle_activities(
    user_input: str,
    deps: TravelDeps,
    ctx: TripContext,
    personality: AgentPersonality,
    observers: list[AgentCallObserver],
) -> None:
    """Step 4: Suggest activities using full context."""
    agent = create_activity_agent(personality)
    prompt = (
        f"Suggest activities in {ctx.destination} near {ctx.hotel_location}, "
        f"suitable for someone arriving at {ctx.arrival_time}."
    )
    output: ActivitiesOutput = await run_agent_with_observers(
        agent, prompt, deps, observers
    )
    ctx.activities = output.activities
    print(f"🎯 Activities:")
    for activity in ctx.activities:
        print(f"   - {activity}")


# ==================================
# MAIN CHAIN EXECUTOR
# ==================================


async def plan_trip(
    user_input: str,
    deps: TravelDeps,
    personality: AgentPersonality = AgentPersonality.PROFESSIONAL,
    observers: Optional[list[AgentCallObserver]] = None,
) -> TripContext:
    """
    Plans a complete trip using the chain of responsibility pattern.

    - Strategy: personality determines agent behavior
    - Chain: handlers execute in sequence, sharing context
    - Observer: all agent calls are logged/monitored
    """
    if observers is None:
        observers = []

    print(f"\n{'='*60}")
    print(f"🌍 TRIP PLANNER - {personality.value.upper()} MODE")
    print(f"{'='*60}")
    print(f"👤 {deps.user_name} from {deps.origin_city}")
    print(f"📝 Request: {user_input}")
    print(f"{'='*60}")

    ctx = TripContext()

    # Chain of handlers - easy to add/remove/reorder
    chain = [
        handle_destination,
        handle_flight,
        handle_hotel,
        handle_activities,
    ]

    for handler in chain:
        await handler(user_input, deps, ctx, personality, observers)

    return ctx


# ==================================
# MAIN EXECUTION
# ==================================


async def main():
    deps = TravelDeps(user_name="Sahil", origin_city="Toronto")
    user_request = "I want a sunny beach vacation somewhere in Europe."

    # Setup observers (Observer Pattern)
    console_logger = ConsoleLogger()
    metrics = MetricsCollector()
    observers = [console_logger, metrics]

    # Run with different strategies (Strategy Pattern)
    # Each strategy uses the same chain (Chain of Responsibility)

    print("\n" + "🔷" * 30)
    print("RUNNING WITH PROFESSIONAL PERSONALITY")
    print("🔷" * 30)
    ctx1 = await plan_trip(
        user_request,
        deps,
        personality=AgentPersonality.PROFESSIONAL,
        observers=observers,
    )

    print("\n" + "🔶" * 30)
    print("RUNNING WITH FUN PERSONALITY")
    print("🔶" * 30)
    ctx2 = await plan_trip(
        user_request,
        deps,
        personality=AgentPersonality.FUN,
        observers=observers,
    )

    print("\n" + "🟢" * 30)
    print("RUNNING WITH BUDGET PERSONALITY")
    print("🟢" * 30)
    ctx3 = await plan_trip(
        user_request,
        deps,
        personality=AgentPersonality.BUDGET,
        observers=observers,
    )

    # Show collected metrics (Observer Pattern benefit)
    metrics.summary()

    # Compare results
    print("\n" + "=" * 60)
    print("📊 COMPARISON OF RESULTS")
    print("=" * 60)
    for label, ctx in [
        ("Professional", ctx1),
        ("Fun", ctx2),
        ("Budget", ctx3),
    ]:
        print(f"\n{label}:")
        print(f"  Destination: {ctx.destination}")
        print(f"  Hotel: {ctx.hotel_name} (${ctx.hotel_price}/night)")


if __name__ == "__main__":
    asyncio.run(main())
