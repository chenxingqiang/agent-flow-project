"""Constants for the agentflow package."""

# Valid communication protocols
VALID_PROTOCOLS = {
    "federated",
    "gossip", 
    "hierarchical",
    "hierarchical_merge",
    None
}

# Valid workflow step types
VALID_STEP_TYPES = {
    "transform",
    "research",
    "document",
    "agent"
}

# Valid workflow strategies
VALID_STRATEGIES = {
    "feature_engineering",
    "outlier_removal",
    "custom",
    "hierarchical",
    "hierarchical_merge",
    "default",
    "federated",
    "gossip",
    "standard",
    "research",
    "document"
} 