import json
import uuid
from typing import Dict
from dotenv import load_dotenv

# Auto-load variables from .env if present
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from core_engine import IRTEngine, KnowledgeGraphManager, AgentOrchestrator

app = FastAPI(title="Adaptive Technical Interviewing Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Graph & Orchestrator Instances
KG_MANAGER = KnowledgeGraphManager("knowledge_graph.json")
ORCHESTRATOR = AgentOrchestrator()

# Session State Store
class InterviewSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.irt = IRTEngine(initial_theta=0.0)
        self.evaluated_nodes: Dict[str, float] = {}  # {node_id: highest_score}
        self.active_node: Dict = None
        self.question_count: int = 0
        self.max_total_questions: int = 15
        self.last_misconceptions = []
        self.history = []  # Detailed Q&A and evaluation log
        
        # Topic Drill-Down Tracking
        self.current_topic_question_count: int = 0
        self.max_questions_per_topic: int = 3

    def start_interview(self) -> Dict:
        eligible = KG_MANAGER.get_eligible_nodes(self.evaluated_nodes)
        self.active_node = KG_MANAGER.select_optimal_next_node(eligible, self.irt)
        self.question_count = 1
        self.current_topic_question_count = 1
        
        initial_question = ORCHESTRATOR.generate_question(
            next_concept_name=self.active_node["name"],
            difficulty_b=self.active_node["difficulty_b"],
            prior_misconceptions=[],
            theta=self.irt.theta.item(),
            is_drilldown=False,
        )
        self.active_question = initial_question
        return {
            "node": self.active_node,
            "question": initial_question,
            "theta": self.irt.theta.item(),
            "se": self.irt.get_standard_error(),
            "evaluated_nodes": self.evaluated_nodes,
        }

    def process_turn(self, candidate_response: str) -> Dict:
        # 1. Evaluate candidate response
        evaluation = ORCHESTRATOR.evaluate_response(
            concept_name=self.active_node["name"],
            difficulty_b=self.active_node["difficulty_b"],
            ground_truth=self.active_node["ground_truth_criteria"],
            candidate_response=candidate_response,
        )

        score = evaluation.aggregate_score
        node_id = self.active_node["node_id"]

        # Log history
        self.history.append({
            "topic_id": node_id,
            "topic_name": self.active_node["name"],
            "difficulty_b": self.active_node["difficulty_b"],
            "question": getattr(self, "active_question", ""),
            "answer": candidate_response,
            "evaluation": evaluation.model_dump(),
            "ground_truth_criteria": self.active_node["ground_truth_criteria"],
            "is_drilldown": (self.current_topic_question_count > 1),
            "drilldown_turn": self.current_topic_question_count,
        })
        
        # Keep best score for prerequisite satisfaction
        prev_score = self.evaluated_nodes.get(node_id, 0.0)
        self.evaluated_nodes[node_id] = max(prev_score, score)
        self.last_misconceptions = evaluation.identified_misconceptions

        # 2. Update IRT Ability Estimate (Theta)
        self.irt.record_response(
            node_id=node_id,
            a=self.active_node["discrimination_a"],
            b=self.active_node["difficulty_b"],
            score=score,
        )
        updated_theta = self.irt.update_theta_newton_raphson()
        updated_se = self.irt.get_standard_error()

        # 3. RULE A: Early Termination if first answer on first topic was extremely wrong / gibberish (< 0.20)
        if self.question_count == 1 and score < 0.20:
            return {
                "interview_complete": True,
                "evaluation": evaluation.model_dump(),
                "final_theta": round(updated_theta, 3),
                "final_se": round(updated_se, 3),
                "summary": self.evaluated_nodes,
                "history": self.history,
                "early_termination_reason": "Interview ended early: Initial response demonstrated severe lack of fundamental technical domain knowledge.",
            }

        # 4. RULE B: DRILL-DOWN LOGIC (Max 3 questions per topic)
        should_drill_down = (score < 0.70) and (self.current_topic_question_count < self.max_questions_per_topic)

        if should_drill_down:
            self.current_topic_question_count += 1
            self.question_count += 1
            drill_question = ORCHESTRATOR.generate_question(
                next_concept_name=self.active_node["name"],
                difficulty_b=self.active_node["difficulty_b"],
                prior_misconceptions=self.last_misconceptions,
                theta=updated_theta,
                is_drilldown=True,
            )
            self.active_question = drill_question
            return {
                "interview_complete": False,
                "evaluation": evaluation.model_dump(),
                "updated_theta": round(updated_theta, 3),
                "updated_se": round(updated_se, 3),
                "next_node": self.active_node,
                "next_question": drill_question,
                "question_number": self.question_count,
                "is_drilldown": True,
                "topic_question_number": self.current_topic_question_count,
                "evaluated_nodes": self.evaluated_nodes,
                "history": self.history,
            }

        # 5. Moving to Next Topic (Candidate satisfied topic OR reached max 3 attempts)
        self.current_topic_question_count = 1

        # Check Global Stopping Criteria
        if self.question_count >= self.max_total_questions or updated_se <= 0.30:
            return {
                "interview_complete": True,
                "evaluation": evaluation.model_dump(),
                "final_theta": round(updated_theta, 3),
                "final_se": round(updated_se, 3),
                "summary": self.evaluated_nodes,
                "history": self.history,
            }

        # 6. Traversal: Select Next Optimal Node
        eligible = KG_MANAGER.get_eligible_nodes(self.evaluated_nodes)
        if not eligible:
            return {
                "interview_complete": True,
                "evaluation": evaluation.model_dump(),
                "final_theta": round(updated_theta, 3),
                "final_se": round(updated_se, 3),
                "summary": self.evaluated_nodes,
                "history": self.history,
            }

        self.active_node = KG_MANAGER.select_optimal_next_node(eligible, self.irt)
        self.question_count += 1

        # Synthesize Next Topic Question
        next_question = ORCHESTRATOR.generate_question(
            next_concept_name=self.active_node["name"],
            difficulty_b=self.active_node["difficulty_b"],
            prior_misconceptions=self.last_misconceptions,
            theta=updated_theta,
            is_drilldown=False,
        )
        self.active_question = next_question

        return {
            "interview_complete": False,
            "evaluation": evaluation.model_dump(),
            "updated_theta": round(updated_theta, 3),
            "updated_se": round(updated_se, 3),
            "next_node": self.active_node,
            "next_question": next_question,
            "question_number": self.question_count,
            "is_drilldown": False,
            "topic_question_number": 1,
            "evaluated_nodes": self.evaluated_nodes,
            "history": self.history,
        }


ACTIVE_SESSIONS: Dict[str, InterviewSession] = {}


@app.websocket("/ws/interview")
async def websocket_interview_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = InterviewSession(session_id)
    ACTIVE_SESSIONS[session_id] = session

    try:
        # Step A: Initialize interview & transmit first question
        try:
            init_payload = session.start_interview()
            await websocket.send_json({
                "event": "INTERVIEW_INITIALIZED",
                "session_id": session_id,
                "question": init_payload["question"],
                "node_name": init_payload["node"]["name"],
                "node_id": init_payload["node"]["node_id"],
                "theta": round(init_payload["theta"], 3),
                "se": round(init_payload["se"], 3),
                "evaluated_nodes": init_payload["evaluated_nodes"],
            })
        except Exception as init_err:
            print(f"Error during interview initialization: {init_err}")
            await websocket.send_json({
                "event": "ERROR",
                "message": f"LLM Generation Failed: {str(init_err)}"
            })
            raise init_err

        # Step B: Main event loop
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("event") == "SUBMIT_ANSWER":
                candidate_text = message.get("text", "")
                result = session.process_turn(candidate_text)

                if result["interview_complete"]:
                    await websocket.send_json({
                        "event": "INTERVIEW_FINISHED",
                        "final_theta": result["final_theta"],
                        "final_se": result["final_se"],
                        "last_evaluation": result["evaluation"],
                        "mastery_summary": result["summary"],
                        "history": result.get("history", []),
                        "early_termination_reason": result.get("early_termination_reason"),
                    })
                    break
                else:
                    await websocket.send_json({
                        "event": "EVALUATION_AND_NEXT_QUESTION",
                        "evaluation": result["evaluation"],
                        "updated_theta": result["updated_theta"],
                        "updated_se": result["updated_se"],
                        "question": result["next_question"],
                        "node_name": result["next_node"]["name"],
                        "node_id": result["next_node"]["node_id"],
                        "question_number": result["question_number"],
                        "is_drilldown": result.get("is_drilldown", False),
                        "topic_question_number": result.get("topic_question_number", 1),
                        "evaluated_nodes": result.get("evaluated_nodes", {}),
                    })

    except WebSocketDisconnect:
        if session_id in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[session_id]


if __name__ == "__main__":
    import uvicorn
    # Requires OPENAI_API_KEY exported in environment
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
