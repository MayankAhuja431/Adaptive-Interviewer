import json
import os
import torch
import networkx as nx
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Auto-load .env
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Priority 1: Groq (if GROQ_API_KEY is present or provided)
# Priority 2: Gemini (if GEMINI_API_KEY is present)
# Priority 3: OpenAI (fallback)
if os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq
elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    from langchain_google_genai import ChatGoogleGenerativeAI
else:
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        pass
    from langchain_openai import ChatOpenAI


# ============================================================================
# 1. PSYCHOMETRIC 2PL-IRT & FISHER INFORMATION ENGINE (PYTORCH)
# ============================================================================

class IRTEngine:
    def __init__(self, initial_theta: float = 0.0, device: str = "cpu"):
        self.device = torch.device(device)
        self.theta = torch.tensor(initial_theta, dtype=torch.float32, device=self.device)
        self.a_params: List[float] = []
        self.b_params: List[float] = []
        self.scores: List[float] = []
        self.visited_nodes: List[str] = []

    def probability_2pl(self, theta: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """P_i(theta) = 1 / (1 + exp(-a_i * (theta - b_i)))"""
        return 1.0 / (1.0 + torch.exp(-a * (theta - b)))

    def fisher_information(self, theta: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """I_i(theta) = a_i^2 * P_i(theta) * (1 - P_i(theta))"""
        p = self.probability_2pl(theta, a, b)
        return (a ** 2) * p * (1.0 - p)

    def record_response(self, node_id: str, a: float, b: float, score: float) -> None:
        self.visited_nodes.append(node_id)
        self.a_params.append(a)
        self.b_params.append(b)
        self.scores.append(max(0.0, min(1.0, score)))

    def update_theta_newton_raphson(self, max_iter: int = 20, tol: float = 1e-4) -> float:
        """Newton-Raphson MLE update: theta_{t+1} = theta_t + (Score / Fisher_Sum)"""
        if not self.scores:
            return float(self.theta.item())

        a_t = torch.tensor(self.a_params, dtype=torch.float32, device=self.device)
        b_t = torch.tensor(self.b_params, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(self.scores, dtype=torch.float32, device=self.device)

        curr_theta = self.theta.clone()
        for _ in range(max_iter):
            p = self.probability_2pl(curr_theta, a_t, b_t)
            first_deriv = torch.sum(a_t * (y_t - p))
            fisher_sum = torch.sum((a_t ** 2) * p * (1.0 - p))

            if fisher_sum.item() < 1e-5:
                break

            step = torch.clamp(first_deriv / fisher_sum, min=-0.75, max=0.75)
            curr_theta += step
            curr_theta = torch.clamp(curr_theta, min=-3.0, max=3.0)

            if torch.abs(step).item() < tol:
                break

        self.theta = curr_theta
        return float(self.theta.item())

    def get_standard_error(self) -> float:
        """SE(theta) = 1 / sqrt(Sum(I_i(theta)))"""
        if not self.scores:
            return 1.50
        a_t = torch.tensor(self.a_params, dtype=torch.float32, device=self.device)
        b_t = torch.tensor(self.b_params, dtype=torch.float32, device=self.device)
        info_sum = torch.sum(self.fisher_information(self.theta, a_t, b_t))
        if info_sum.item() <= 1e-5:
            return 1.50
        return float((1.0 / torch.sqrt(info_sum)).item())


# ============================================================================
# 2. NETWORKX DAG PREREQUISITE ENGINE & GRAPH TRAVERSAL
# ============================================================================

class KnowledgeGraphManager:
    def __init__(self, json_filepath: str):
        self.graph = nx.DiGraph()
        self.load_graph(json_filepath)

    def load_graph(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)

        for item in nodes_data:
            self.graph.add_node(
                item["node_id"],
                name=item["name"],
                topic=item["topic"],
                difficulty_b=float(item["difficulty_b"]),
                discrimination_a=float(item["discrimination_a"]),
                ground_truth_criteria=item["ground_truth_criteria"],
            )

        for item in nodes_data:
            for prereq in item.get("prerequisites", []):
                # Edge points from Prerequisite -> Child Concept
                self.graph.add_edge(prereq, item["node_id"])

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Invalid Knowledge Graph: Cycles detected in prerequisite chain!")

    def get_eligible_nodes(self, evaluated_nodes: Dict[str, float]) -> List[Dict]:
        """
        Returns unvisited nodes whose prerequisites are all satisfied (score >= 0.50).
        """
        eligible = []
        for node_id in self.graph.nodes:
            if node_id in evaluated_nodes:
                continue

            prereqs = list(self.graph.predecessors(node_id))
            prereqs_satisfied = True
            for p in prereqs:
                if p not in evaluated_nodes or evaluated_nodes[p] < 0.50:
                    prereqs_satisfied = False
                    break

            if prereqs_satisfied:
                node_attr = self.graph.nodes[node_id].copy()
                node_attr["node_id"] = node_id
                eligible.append(node_attr)
        return eligible

    def select_optimal_next_node(
        self, eligible_nodes: List[Dict], irt_engine: IRTEngine
    ) -> Optional[Dict]:
        """Chooses the candidate node that maximizes Fisher Information at current theta."""
        if not eligible_nodes:
            return None

        node_a = torch.tensor([n["discrimination_a"] for n in eligible_nodes], dtype=torch.float32)
        node_b = torch.tensor([n["difficulty_b"] for n in eligible_nodes], dtype=torch.float32)

        fisher_vals = irt_engine.fisher_information(irt_engine.theta, node_a, node_b)
        best_idx = int(torch.argmax(fisher_vals).item())
        selected = eligible_nodes[best_idx].copy()
        selected["expected_fisher_info"] = round(float(fisher_vals[best_idx].item()), 4)
        return selected


# ============================================================================
# 3. DUAL-AGENT PIPELINE: EVALUATOR (G-EVAL) & INTERVIEWER (LANGCHAIN)
# ============================================================================

class EvaluatorSchema(BaseModel):
    conceptual_accuracy: float = Field(ge=0.0, le=1.0, description="Accuracy of concepts")
    technical_depth: float = Field(ge=0.0, le=1.0, description="Completeness and depth")
    reasoning_ability: float = Field(ge=0.0, le=1.0, description="Cause-and-effect reasoning")
    clarity: float = Field(ge=0.0, le=1.0, description="Structure and technical terminology")
    aggregate_score: float = Field(ge=0.0, le=1.0, description="Weighted composite score")
    identified_misconceptions: List[str] = Field(description="Explicit conceptual errors found")
    candidate_strengths: List[str] = Field(description="Points candidate answered accurately")
    prerequisite_status: str = Field(description="SATISFIED, PARTIAL, or FAILED")


class AgentOrchestrator:
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.0):
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if groq_key:
            clean_key = groq_key.strip().strip("'\"")
            chosen_model = model_name or "openai/gpt-oss-120b"
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(api_key=clean_key, model=chosen_model, temperature=temperature)
        elif gemini_key:
            clean_key = gemini_key.strip().strip("'\"")
            os.environ["GOOGLE_API_KEY"] = clean_key
            os.environ["GEMINI_API_KEY"] = clean_key
            chosen_model = model_name or "gemini-2.0-flash"
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=chosen_model,
                api_key=clean_key,
                temperature=temperature,
                max_retries=2,
            )
        else:
            chosen_model = model_name or "gpt-4o-mini"
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=chosen_model, temperature=temperature)

        self.parser = PydanticOutputParser(pydantic_object=EvaluatorSchema)

        self.eval_prompt = PromptTemplate(
            template=(
                "You are an encouraging yet technically sound interviewer evaluating an engineering candidate's verbal or typed response.\n"
                "Concept Node: {concept_name}\n"
                "Target Difficulty (b): {difficulty_b}\n\n"
                "Ground-Truth Required Criteria:\n{ground_truth_criteria}\n\n"
                "Candidate Verbal Transcript:\n\"{candidate_response}\"\n\n"
                "Instructions for Fair & Encouraging Evaluation:\n"
                "1. Award credit for core understanding and conceptual grasp, even if described in plain everyday words without exact textbook jargon.\n"
                "2. If the candidate correctly conveys the fundamental idea behind any ground-truth criterion, award solid credit for accuracy and depth.\n"
                "3. Calculate individual scores between 0.0 and 1.0:\n"
                "   - conceptual_accuracy: Did they get the core idea right? (Give 0.70+ for good fundamental grasp)\n"
                "   - technical_depth: Did they mention relevant mechanics or properties? (Give 0.60+ for reasonable detail)\n"
                "   - reasoning_ability: Logical cause-and-effect reasoning.\n"
                "   - clarity: Clear communication.\n"
                "   aggregate_score = (0.40 * conceptual_accuracy) + (0.25 * technical_depth) + (0.20 * reasoning_ability) + (0.15 * clarity)\n"
                "4. Status determination:\n"
                "   - If aggregate_score >= 0.50 set prerequisite_status to 'SATISFIED'\n"
                "   - If between 0.35 and 0.50 set 'PARTIAL'\n"
                "   - Only if aggregate_score < 0.35 set 'FAILED'\n\n"
                "{format_instructions}"
            ),
            input_variables=["concept_name", "difficulty_b", "ground_truth_criteria", "candidate_response"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        self.interview_prompt = PromptTemplate(
            template=(
                "You are an encouraging yet rigorous senior software engineer conducting a technical interview.\n"
                "Target Concept: {next_concept_name}\n"
                "Difficulty Level: {difficulty_b} (Scale: -3.0 Easy to +3.0 Advanced)\n"
                "Candidate Ability (Theta): {theta:.2f}\n"
                "Mode: {mode_instruction}\n\n"
                "Task: Formulate a single, concise open-ended technical question testing this concept.\n"
                "Do not lecture, explain the answer, or give away solutions. Keep your question strictly under 3 sentences."
            ),
            input_variables=["next_concept_name", "difficulty_b", "mode_instruction", "theta"],
        )

    def evaluate_response(
        self, concept_name: str, difficulty_b: float, ground_truth: List[str], candidate_response: str
    ) -> EvaluatorSchema:
        prompt_val = self.eval_prompt.format_prompt(
            concept_name=concept_name,
            difficulty_b=difficulty_b,
            ground_truth_criteria="\n- ".join(ground_truth),
            candidate_response=candidate_response,
        )
        response = self.llm.invoke(prompt_val.to_messages())
        return self.parser.parse(response.content)

    def generate_question(
        self, next_concept_name: str, difficulty_b: float, prior_misconceptions: List[str], theta: float, is_drilldown: bool = False
    ) -> str:
        if is_drilldown:
            misconceptions_str = "; ".join(prior_misconceptions) if prior_misconceptions else "missing key depth"
            mode = (
                f"DRILL-DOWN FOLLOW-UP on the same topic: The candidate gave a partial answer with gaps/misconceptions: ({misconceptions_str}). "
                "Ask a focused follow-up question digging deeper into the missing technical mechanism without repeating the previous question."
            )
        else:
            misconceptions_str = ", ".join(prior_misconceptions) if prior_misconceptions else "None"
            mode = f"NEW TOPIC: Smoothly transition to test this new topic. Prior misconceptions to note: {misconceptions_str}."

        prompt_val = self.interview_prompt.format_prompt(
            next_concept_name=next_concept_name,
            difficulty_b=difficulty_b,
            mode_instruction=mode,
            theta=theta,
        )
        response = self.llm.invoke(prompt_val.to_messages())
        return response.content.strip()
