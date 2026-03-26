"""
Dependency Graph Agent for cross-repository dependency tracking.
Detects and models dependencies across multiple repositories.
Part of LangChain DeepAgents harness implementation.
"""

import re
import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class DependencyGraphAgent:
    """
    Models cross-repository dependencies and predicts risk propagation.

    Capabilities:
    - Parse code imports (Python, Node.js, Go)
    - Extract issue cross-references
    - Build dependency DAG
    - Compute dependency chains
    - Predict risk propagation across repos
    """

    def __init__(self, tool_registry):
        """Initialize with tool registry."""
        self.tool_registry = tool_registry
        self.dependencies = defaultdict(set)  # repo -> set of dependent repos
        self.dependents = defaultdict(set)    # repo -> set of repos that depend on it
        self.issue_refs = defaultdict(list)   # repo -> list of cross-repo issue refs
        self.import_graph = {}                # detailed import graph
        self.dependency_chains = {}           # cached chains for each repo

    def execute(self, state) -> object:
        """
        Execute dependency graph analysis.

        Args:
            state: OrchestratorState with repositories and code data

        Returns:
            Updated state with dependency information
        """
        logger.info("=== Dependency Graph Agent Starting ===")
        state.current_agent = "dependency_graph"

        try:
            # Parse imports from code
            logger.info(f"  • Parsing imports for {len(state.repositories)} repos")
            self._parse_imports(state)

            # Extract issue cross-references
            logger.info(f"  • Extracting cross-repo issue references")
            self._extract_issue_refs(state)

            # Build dependency graph
            logger.info(f"  • Building dependency DAG")
            dep_graph = self._build_dependency_graph()

            # Compute longest chains
            logger.info(f"  • Computing dependency chains")
            chains = self._compute_dependency_chains()

            # Predict risk propagation
            logger.info(f"  • Analyzing risk propagation")
            propagation = self._predict_risk_propagation(state)

            # Store results in state
            state.dependency_graph = {
                "nodes": list(state.repositories),
                "edges": dep_graph,
                "dependency_chains": chains,
                "risk_propagation": propagation,
                "import_graph": self.import_graph,
                "issue_references": dict(self.issue_refs),
            }

            state.execution_logs.append(
                f"[dependency_graph] Analyzed {len(state.repositories)} repos, "
                f"found {len(dep_graph)} dependencies"
            )
            logger.info(f"  ✓ Dependency graph computed")

        except Exception as e:
            state.errors.append(f"Dependency graph error: {str(e)}")
            logger.error(f"Dependency graph error: {e}")
            # Gracefully degrade - continue without dependency analysis
            state.dependency_graph = {
                "nodes": list(state.repositories),
                "edges": [],
                "dependency_chains": {},
                "risk_propagation": {},
                "import_graph": {},
                "issue_references": {},
            }

        return state

    def _parse_imports(self, state):
        """
        Parse code imports to detect repository dependencies.
        Supports Python, Node.js, and Go.
        """
        # Python import patterns
        python_patterns = [
            r'^\s*import\s+(\w+)',
            r'^\s*from\s+(\w+)',
        ]

        # JavaScript/Node patterns
        js_patterns = [
            r"require\(['\"]([^'\"]+)['\"]\)",
            r"import\s+.*from\s+['\"]([^'\"]+)['\"]",
        ]

        # Go patterns
        go_patterns = [
            r'import\s+["\']([^"\']+)["\']',
        ]

        for repo in state.repositories:
            imports = set()

            # Simulate code parsing (would be real AST parsing in production)
            if "code" in state and repo in state.code:
                code = state.code[repo]

                # Try Python patterns
                for pattern in python_patterns:
                    matches = re.findall(pattern, code, re.MULTILINE)
                    imports.update(matches)

                # Try JS patterns
                for pattern in js_patterns:
                    matches = re.findall(pattern, code)
                    imports.update(matches)

                # Try Go patterns
                for pattern in go_patterns:
                    matches = re.findall(pattern, code)
                    imports.update(matches)

            # Build import graph
            self.import_graph[repo] = list(imports)

            # Link to other repos if imports match
            for other_repo in state.repositories:
                if repo != other_repo:
                    repo_name = other_repo.split('/')[-1]
                    if any(repo_name in imp for imp in imports):
                        self.dependencies[repo].add(other_repo)
                        self.dependents[other_repo].add(repo)

    def _extract_issue_refs(self, state):
        """
        Extract cross-repository issue references from issue/PR bodies.
        Format: Org/Repo#123
        """
        ref_pattern = r'([A-Za-z0-9\-]+)/([A-Za-z0-9\-_]+)#(\d+)'

        for repo in state.repositories:
            refs = []

            # Check issues
            if hasattr(state, 'github_issues') and state.github_issues:
                for issue in state.github_issues:
                    if hasattr(issue, 'body') and issue.body:
                        matches = re.findall(ref_pattern, issue.body)
                        for org, repo_name, issue_num in matches:
                            ref_repo = f"{org}/{repo_name}"
                            if ref_repo != repo and ref_repo in state.repositories:
                                refs.append({
                                    "referenced_repo": ref_repo,
                                    "issue_number": issue_num,
                                    "source_issue": issue.number if hasattr(issue, 'number') else None,
                                    "is_blocker": any(
                                        label in (issue.labels if hasattr(issue, 'labels') else [])
                                        for label in ["blocker", "blocked", "depends-on", "dependency"]
                                    )
                                })

            self.issue_refs[repo] = refs

    def _build_dependency_graph(self) -> List[Dict]:
        """
        Build dependency graph edges from parsed imports and issue refs.
        Returns list of edges: [{"source": repo_a, "target": repo_b, "type": "import|issue"}]
        """
        edges = []

        # Add import dependencies
        for repo, deps in self.dependencies.items():
            for dep in deps:
                edges.append({
                    "source": repo,
                    "target": dep,
                    "type": "import",
                    "label": "imports from"
                })

        # Add issue dependencies
        for repo, refs in self.issue_refs.items():
            for ref in refs:
                edges.append({
                    "source": repo,
                    "target": ref["referenced_repo"],
                    "type": "issue",
                    "label": "depends on issue",
                    "issue": ref["issue_number"],
                    "is_blocker": ref["is_blocker"]
                })

        return edges

    def _compute_dependency_chains(self) -> Dict[str, List[str]]:
        """
        Compute longest dependency chains from each repo.
        Returns dict mapping repos to their longest dependency paths.
        """
        chains = {}

        for repo in self.dependencies.keys():
            # BFS to find longest path
            queue = deque([(repo, [repo])])
            longest_path = [repo]

            visited = {repo}

            while queue:
                current, path = queue.popleft()

                if len(path) > len(longest_path):
                    longest_path = path

                for dependent in self.dependents.get(current, set()):
                    if dependent not in visited:
                        visited.add(dependent)
                        queue.append((dependent, path + [dependent]))

            chains[repo] = longest_path

        return chains

    def _predict_risk_propagation(self, state) -> Dict[str, float]:
        """
        Predict how delays/risks in one repo propagate to others.
        Returns risk propagation scores (0-1) for each repo.
        """
        propagation = {}

        for repo in state.repositories:
            # Risk score = (number of dependents + longest chain length) / repos
            num_dependents = len(self.dependents.get(repo, set()))
            chain_length = len(self.dependency_chains.get(repo, [repo]))
            total_repos = len(state.repositories)

            # If this repo is delayed, how many others are affected?
            # score = (direct_dependents + direct_dependencies) / total
            impact_score = (num_dependents + len(self.dependencies.get(repo, set()))) / max(total_repos, 1)

            # Chain multiplier: repos in long chains are more critical
            chain_multiplier = min(chain_length / max(total_repos, 1), 1.0)

            # Combined score
            propagation_score = min(impact_score * (1 + chain_multiplier), 1.0)
            propagation[repo] = propagation_score

        return propagation
