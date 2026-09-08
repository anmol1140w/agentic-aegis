"""AEGIS sovereign orchestration primitives.

The package is intentionally provider- and framework-neutral.  It contains
the contracts which keep routing, tools, workflows, governance, retrieval,
memory and evaluation interoperable.
"""

from .contracts import ExecutionPlan, PlanValidator, WorkflowRegistry
from .governance import AuditChain, PolicyEngine, Principal, RBAC
from .retrieval import HybridKnowledgeStore
from .sovereign import (A2AGateway, A2AMessage, AgentTransport, AuditAnchorBackend,
                        BlockchainAuditLedger, DeploymentBackend, DeterministicCapabilityRouter,
                        DSPyProgram, EvidenceVerifier, ExecutionIsolationBackend, FederatedInference,
                        InferenceBackend, KubernetesPlanner, OptimizationStrategy, RLRoutingStrategy, RiskBasedApproval,
                        RoutingStrategy, SovereignExecutor, TEEVerifier, VisualWorkflowEditor)

__all__ = [
    "AuditChain", "ExecutionPlan", "HybridKnowledgeStore", "PlanValidator",
    "PolicyEngine", "Principal", "RBAC", "WorkflowRegistry", "A2AGateway",
    "A2AMessage", "AgentTransport", "AuditAnchorBackend", "BlockchainAuditLedger",
    "DeploymentBackend", "DeterministicCapabilityRouter", "DSPyProgram", "EvidenceVerifier",
    "ExecutionIsolationBackend", "FederatedInference", "InferenceBackend", "KubernetesPlanner", "OptimizationStrategy",
    "RLRoutingStrategy", "RiskBasedApproval", "RoutingStrategy", "SovereignExecutor", "TEEVerifier",
    "VisualWorkflowEditor",
]
