import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent

load_dotenv()

# Visibility when calling multiple agents
# Logging code - observerable agent with structured info - inspect whats happening
# Example below with single agent

# ----------------------------------
# Observer Interface
# ----------------------------------

# ----------------------------------
# Dependencies and Output Schema
# ----------------------------------

@dataclass
class TravelDeps:
    user_name: str
    origin_city: str


class TravelResponse(BaseModel):
    destination: str
    message: str

# any observer must implement notify() with these parameters.
class AgentCallObserver(Protocol):
    def notify(
        self,
        agent_name: str,
        prompt: str,
        deps: TravelDeps,
        output: BaseModel,
        duration: float,
    ) -> None:
        ...

# Also send to logging system, monitoring system, etc.
class ConsoleLogger(AgentCallObserver):
    def notify(
        self,
        agent_name: str,
        prompt: str,
        deps: TravelDeps,
        output: BaseModel,
        duration: float,
    ) -> None:
        print("\n📋 Agent Call Log")
        print(f"Agent: {agent_name}")
        print(f"Prompt: {prompt}")
        print(f"User: {deps.user_name}, Origin: {deps.origin_city}")
        print(f"Output: {output.model_dump()}")
        print(f"Duration: {duration:.2f}s")


# ----------------------------------
# Wrapper to run agent with observers
# ----------------------------------
# Wrapper Function
# How It Works                                                                                                                                                                                                      
                                                                                                                                                                                                                    
#   ┌─────────────────┐                                                                                                                                                                                               
#   │  run_with_      │                                                                                                                                                                                               
#   │  observers()    │                                                                                                                                                                                               
#   └────────┬────────┘                                                                                                                                                                                               
#            │                                                                                                                                                                                                        
#            ▼                                                                                                                                                                                                        
#   ┌─────────────────┐                                                                                                                                                                                               
#   │  agent.run()    │ ──► actual LLM call                                                                                                                                                                           
#   └────────┬────────┘                                                                                                                                                                                               
#            │                                                                                                                                                                                                        
#            ▼                                                                                                                                                                                                        
#       ┌────┴────┐                                                                                                                                                                                                   
#       │ for each│                                                                                                                                                                                                   
#       │ observer│                                                                                                                                                                                                   
#       └────┬────┘                                                                                                                                                                                                   
#            │                                                                                                                                                                                                        
#       ┌────┴────────────────┐                                                                                                                                                                                       
#       ▼                     ▼                                                                                                                                                                                       
#   ┌──────────┐       ┌──────────────┐                                                                                                                                                                               
#   │Console   │       │ Other        │                                                                                                                                                                               
#   │Logger    │       │ Observers... │                                                                                                                                                                               
#   └──────────┘       └──────────────┘ 
async def run_with_observers(
    *,
    agent: Agent[TravelDeps, BaseModel],
    prompt: str,
    deps: TravelDeps,
    observers: list[AgentCallObserver],
) -> TravelResponse:
    start = time.perf_counter() # 1. Start timer
    result = await agent.run(prompt, deps=deps) # 2. Run agent 
    end = time.perf_counter() # 3. Calculate duration
    duration = end - start 

    for observer in observers: # 4. Notify ALL observers 
        observer.notify(
            agent_name=agent.name or "Unnamed Agent",
            prompt=prompt,
            deps=deps,
            output=result.output,
            duration=duration,
        )

    return result.output





# ----------------------------------
# Agent Definition
# ----------------------------------

travel_agent = Agent(
    "openai:gpt-4o",
    name="TravelAgent",
    deps_type=TravelDeps,
    output_type=TravelResponse,
    system_prompt="You are a friendly travel assistant. Recommend a destination based on user preferences.",
)

  # ----------------------------------                                                                                                                                                                              
  # Flight Booking Schema                                                                                                                                                                                           
  # ----------------------------------                                                                                                                                                                              
                                                                                                                                                                                                                    
class FlightBookingResponse(BaseModel):                                                                                                                                                                           
    departure_city: str                                                                                                                                                                                           
    arrival_city: str                                                                                                                                                                                             
    departure_time: str                                                                                                                                                                                           
    arrival_time: str                                                                                                                                                                                             
    airline: str                                                                                                                                                                                                  
    price_usd: int                                                                                                                                                                                                
                                                                                                                                                                                                                    
                                                                                                                                                                                                                    
  # ----------------------------------                                                                                                                                                                              
  # Flight Booking Agent                                                                                                                                                                                            
  # ----------------------------------                                                                                                                                                                              
                                                                                                                                                                                                                    
flight_agent = Agent(                                                                                                                                                                                             
    "openai:gpt-4o",                                                                                                                                                                                              
    name="FlightBookingAgent",                                                                                                                                                                                    
    deps_type=TravelDeps,                                                                                                                                                                                         
    output_type=FlightBookingResponse,                                                                                                                                                                            
    system_prompt="You are a flight booking assistant. Find the best flight based on user preferences and origin city.",                                                                                          
)  

# ----------------------------------
# Main Program
# ----------------------------------

async def main():
    deps = TravelDeps(user_name="Sahil", origin_city="Toronto")
    observers = [ConsoleLogger()]  # shared observers for all agents 

    prompt = "I want to escape to a cozy place in the mountains for the weekend."
    travel_output = await run_with_observers( # call with any agent and get logging
        agent=travel_agent,
        prompt=prompt,
        deps=deps,
        observers=observers,
    )

    print(f"\n🤖 Travel Agent says: {travel_output.message}")
    print(f"📍 Destination Suggested: {travel_output.destination}")

    # Step 2: Book flight to that destination                                                                                                                                                                     
    flight_prompt = f"Book a flight from {deps.origin_city} to {travel_output.destination} for this weekend."                                                                                                     
    flight_output = await run_with_observers(                                                                                                                                                                     
        agent=flight_agent,                                                                                                                                                                                       
        prompt=flight_prompt,                                                                                                                                                                                     
        deps=deps,                                                                                                                                                                                                
        observers=observers,  # same observers = same logging                                                                                                                                                     
    )                                                                                                                                                                                                             
    print(f"\n✈️ Flight: {flight_output.airline} - ${flight_output.price_usd}")    

if __name__ == "__main__":
    asyncio.run(main())