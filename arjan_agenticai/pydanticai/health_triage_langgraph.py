"""
Health Triage System using LangGraph Framework

This script demonstrates a multi-stage health triage workflow using LangGraph's
StateGraph for orchestration. The workflow includes:
1. Patient intake and symptom collection
2. Vitals retrieval from database
3. LLM-powered symptom assessment
4. Conditional routing based on urgency
5. Final triage recommendation generation

Workflow Graph:
    intake -> vitals_check -> assessment -> [emergency_response | standard_care] -> output
"""

from dataclasses import dataclass
from typing import Any, Literal
import operator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

load_dotenv()


##################################################################
# Mock Database
##################################################################

@dataclass
class Patient:
    id: int
    name: str
    age: int
    vitals: dict[str, Any]


PATIENT_DB: dict[int, Patient] = {
    42: Patient(
        id=42,
        name="Bob Johnson",
        age=45,
        vitals={"heart_rate": 72, "blood_pressure": "120/80", "temperature": 98.6, "oxygen_saturation": 98}
    ),
    43: Patient(
        id=43,
        name="Jane Smith",
        age=32,
        vitals={"heart_rate": 65, "blood_pressure": "110/70", "temperature": 98.4, "oxygen_saturation": 99}
    ),
    44: Patient(
        id=44,
        name="Jane Smith",
        age=32,
        vitals={"heart_rate": 145, "blood_pressure": "160/110", "temperature": 101.4, "oxygen_saturation": 94}
    )
}


class DatabaseConn:
    def get_patient(self, patient_id: int) -> Patient | None:
        return PATIENT_DB.get(patient_id)

    def get_patient_name(self, patient_id: int) -> str:
        patient = PATIENT_DB.get(patient_id)
        return patient.name if patient else "Unknown Patient"

    def get_vitals(self, patient_id: int) -> dict[str, Any]:
        patient = PATIENT_DB.get(patient_id)
        return patient.vitals if patient else {}


##################################################################
# Output Schema
##################################################################

class TriageOutput(BaseModel):
    response_text: str = Field(description="Message to the patient")
    escalate: bool = Field(description="Should escalate to a human nurse")
    urgency: int = Field(description="Urgency level from 0 to 10", ge=0, le=10)


##################################################################
# State Definition
##################################################################

class TriageState(TypedDict):
    patient_id: int
    symptoms: str
    patient_name: str
    age: int
    vitals: dict[str, Any]
    urgency_level: int
    assessment: str
    messages: Annotated[list[str], operator.add]
    final_output: TriageOutput | None


##################################################################
# LLM Setup
##################################################################

llm = ChatOpenAI(model="gpt-4o", temperature=0)


##################################################################
# Graph Nodes
##################################################################

def intake_node(state: TriageState) -> dict:
    """
    Intake node: Retrieves patient information from database.
    """
    db = DatabaseConn()
    patient = db.get_patient(state["patient_id"])

    if not patient:
        return {
            "patient_name": "Unknown",
            "age": 0,
            "messages": [f"[Intake] Patient ID {state['patient_id']} not found in database."]
        }

    return {
        "patient_name": patient.name,
        "age": patient.age,
        "messages": [f"[Intake] Patient {patient.name} (ID: {patient.id}) checked in with symptoms: {state['symptoms']}"]
    }


def vitals_check_node(state: TriageState) -> dict:
    """
    Vitals check node: Retrieves latest vital signs from database.
    """
    db = DatabaseConn()
    vitals = db.get_vitals(state["patient_id"])

    if not vitals:
        return {
            "vitals": {},
            "messages": ["[Vitals] No vitals available for this patient."]
        }

    vitals_summary = ", ".join(f"{k}: {v}" for k, v in vitals.items())
    return {
        "vitals": vitals,
        "messages": [f"[Vitals] Retrieved vitals: {vitals_summary}"]
    }


def assessment_node(state: TriageState) -> dict:
    """
    Assessment node: Uses LLM to analyze symptoms, vitals, and medical history.
    Returns an urgency level (0-10) and assessment text.
    """
    prompt = f"""You are a medical triage assistant. Analyze the following patient information and provide an urgency assessment.

Patient: {state['patient_name']}
Age: {state['age']}
Reported Symptoms: {state['symptoms']}
Current Vitals: {state['vitals']}

Based on this information:
1. Assess the urgency level from 0 (routine) to 10 (life-threatening emergency)
2. Provide a brief clinical assessment

Respond in this exact format:
URGENCY: [number 0-10]
ASSESSMENT: [your clinical assessment]
"""

    response = llm.invoke(prompt)
    content = response.content

    # Parse urgency and assessment from response
    urgency = 5  # default
    assessment = content

    for line in content.split("\n"):
        if line.startswith("URGENCY:"):
            try:
                urgency = int(line.replace("URGENCY:", "").strip())
                urgency = max(0, min(10, urgency))
            except ValueError:
                pass
        elif line.startswith("ASSESSMENT:"):
            assessment = line.replace("ASSESSMENT:", "").strip()

    return {
        "urgency_level": urgency,
        "assessment": assessment,
        "messages": [f"[Assessment] Urgency level determined: {urgency}/10"]
    }


def output_node(state: TriageState) -> dict:
    """
    Output node: Creates the final TriageOutput from assessment results.
    """
    output = TriageOutput(
        response_text=state["assessment"],
        escalate=state["urgency_level"] >= 7,
        urgency=state["urgency_level"]
    )

    return {
        "final_output": output,
        "messages": [f"[Output] Triage complete for {state['patient_name']}. Urgency: {state['urgency_level']}/10"]
    }



##################################################################
# Build the Graph
##################################################################

def build_triage_graph() -> StateGraph:
    """
    Constructs the triage workflow graph.

    Graph structure:
        START -> intake -> vitals_check -> assessment -> [route_by_urgency]
            -> emergency_response -> output -> END
            -> standard_care -> output -> END
    """
    graph = StateGraph(TriageState)

    # Add nodes
    graph.add_node("intake", intake_node)
    graph.add_node("vitals_check", vitals_check_node)
    graph.add_node("assessment", assessment_node)
    graph.add_node("output", output_node)

    # Add edges
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "vitals_check")
    graph.add_edge("vitals_check", "assessment")
    graph.add_edge("assessment", "output")
    graph.add_edge("output", END)

    return graph.compile()


##################################################################
# Main Execution
##################################################################

def run_triage(patient_id: int, symptoms: str) -> TriageOutput:
    """
    Runs the triage workflow for a patient.

    Args:
        patient_id: The patient's database ID
        symptoms: The patient's reported symptoms

    Returns:
        TriageOutput with response, escalation status, urgency, and recommendations
    """
    graph = build_triage_graph()

    initial_state: TriageState = {
        "patient_id": patient_id,
        "symptoms": symptoms,
        "patient_name": "",
        "age": 0,
        "vitals": {},
        "urgency_level": 0,
        "assessment": "",
        "messages": [],
        "final_output": None,
    }

    result = graph.invoke(initial_state)

    # Print workflow messages
    print("\n" + "=" * 60)
    print("TRIAGE WORKFLOW LOG")
    print("=" * 60)
    for msg in result["messages"]:
        print(msg)
    print("=" * 60 + "\n")

    return result["final_output"]


def main() -> None:
    # Test Case 1: High urgency - chest pain
    print("\n" + "#" * 60)
    print("TEST CASE 1: Emergency Symptoms")
    print("#" * 60)

    output1 = run_triage(
        patient_id=44,  # Bob Johnson - has heart disease history
        symptoms="I have severe chest pain radiating to my left arm, shortness of breath, and I'm feeling dizzy."
    )
    print("TRIAGE RESULT:")
    print(f"  Response: {output1.response_text}...")
    print(f"  Escalate: {output1.escalate}")
    print(f"  Urgency: {output1.urgency}/10")

    # Test Case 2: Low urgency - mild symptoms
    print("\n" + "#" * 60)
    print("TEST CASE 2: Routine Symptoms")
    print("#" * 60)

    output2 = run_triage(
        patient_id=43,  # Jane Smith - healthy patient
        symptoms="I have a mild headache since yesterday and slight fatigue."
    )
    print("TRIAGE RESULT:")
    print(f"  Response: {output2.response_text[:200]}...")
    print(f"  Escalate: {output2.escalate}")
    print(f"  Urgency: {output2.urgency}/10")

    # Test Case 3: Medium urgency
    print("\n" + "#" * 60)
    print("TEST CASE 3: Moderate Symptoms")
    print("#" * 60)

    output3 = run_triage(
        patient_id=42,  # John Doe - has diabetes and hypertension
        symptoms="I have persistent fever of 101F for two days, body aches, and a sore throat."
    )
    print("TRIAGE RESULT:")
    print(f"  Response: {output3.response_text[:200]}...")
    print(f"  Escalate: {output3.escalate}")
    print(f"  Urgency: {output3.urgency}/10")


if __name__ == "__main__":
    main()
