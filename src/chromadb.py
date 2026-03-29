"""
ChromaDB ingestion and RAG retrieval for Sprint Intelligence.

Single-file module handling:
1. Bulk ingestion of sprint JSON files (real + synthetic) into ChromaDB
2. Rich document embedding (sprint summaries, issues, PRs, commits)
3. Similarity search with full context retrieval for RAG
4. Evidence citation generation (GitHub URLs for commits/issues/PRs)

Usage:
    # Ingest all sprint data
    db = SprintChromaDB()
    db.ingest_all_data()

    # Query during agent inference
    rag = db.query_similar_sprints(owner="Mintplex-Labs", repo="anything-llm",
                                     sprint_id="sprint_005", features={...})
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import chromadb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

# ChromaDB collection names
COLLECTION_SPRINTS = "sprint_documents"

# Max metadata value length ChromaDB allows (for safety)
_MAX_META_STR = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_meta(value: Any) -> Any:
    """Coerce a value into a ChromaDB-safe metadata type (str|int|float|bool)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return json.dumps(value)[:_MAX_META_STR]
    if isinstance(value, dict):
        return json.dumps(value)[:_MAX_META_STR]
    if value is None:
        return ""
    return str(value)[:_MAX_META_STR]


def _github_url(owner: str, repo: str, kind: str, number_or_sha: Any) -> str:
    """Build a GitHub URL for an issue, PR, or commit."""
    base = f"https://github.com/{owner}/{repo}"
    if kind == "issue":
        return f"{base}/issues/{number_or_sha}"
    if kind == "pr":
        return f"{base}/pull/{number_or_sha}"
    if kind == "commit":
        return f"{base}/commit/{number_or_sha}"
    return base


def _extract_owner_repo(repo_slug: str) -> tuple[str, str]:
    """Extract owner and repo name from 'owner/repo' slug."""
    parts = repo_slug.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", repo_slug


# ---------------------------------------------------------------------------
# Document builders – turn raw sprint JSON into embeddable documents
# ---------------------------------------------------------------------------

def _build_sprint_summary_doc(sprint: dict, owner: str, repo: str) -> dict:
    """Build a rich text document for a sprint summary."""
    sid = sprint["sprint_id"]
    m = sprint.get("metrics", {})
    risk = sprint.get("risk_label", {})

    content_parts = [
        f"Sprint {sid} for {owner}/{repo}",
        f"Period: {sprint.get('start_date', '')} to {sprint.get('end_date', '')}",
        f"Duration: {m.get('days_span', 0)} days",
        "",
        "Activity:",
        f"  Issues: {m.get('total_issues', 0)} total, {m.get('closed_issues', 0)} closed (resolution rate: {m.get('issue_resolution_rate', 0):.2%})",
        f"  PRs: {m.get('total_prs', 0)} total, {m.get('merged_prs', 0)} merged (merge rate: {m.get('pr_merge_rate', 0):.2%})",
        f"  Commits: {m.get('total_commits', 0)} (frequency: {m.get('commit_frequency', 0):.2f}/day)",
        "",
        "Code:",
        f"  Changes: +{m.get('total_additions', 0)} -{m.get('total_deletions', 0)} across {m.get('files_changed', 0)} files",
        f"  Avg PR size: {m.get('avg_pr_size', 0):.0f} lines, concentration: {m.get('code_concentration', 0):.2f}",
        "",
        "Risk indicators:",
        f"  Stalled issues: {m.get('stalled_issues', 0)}, Unreviewed PRs: {m.get('unreviewed_prs', 0)}",
        f"  Abandoned PRs: {m.get('abandoned_prs', 0)}, Long-open issues: {m.get('long_open_issues', 0)}",
        "",
        "Team:",
        f"  Unique authors: {m.get('unique_authors', 0)}, participation: {m.get('author_participation', 0):.2f}",
        "",
        f"Risk score: {risk.get('risk_score', 0):.2f}, at-risk: {risk.get('is_at_risk', False)}",
    ]
    if risk.get("risk_factors"):
        content_parts.append(f"Risk factors: {', '.join(str(r) for r in risk['risk_factors'] if r)}")

    metadata = {
        "sprint_id": sid,
        "owner": owner,
        "repo": repo,
        "repo_full": f"{owner}/{repo}",
        "type": "sprint_summary",
        "start_date": _safe_meta(sprint.get("start_date", "")),
        "end_date": _safe_meta(sprint.get("end_date", "")),
        "risk_score": float(risk.get("risk_score", 0)),
        "is_at_risk": bool(risk.get("is_at_risk", False)),
        "total_issues": int(m.get("total_issues", 0)),
        "total_prs": int(m.get("total_prs", 0)),
        "total_commits": int(m.get("total_commits", 0)),
        "closed_issues": int(m.get("closed_issues", 0)),
        "merged_prs": int(m.get("merged_prs", 0)),
        "issue_resolution_rate": float(m.get("issue_resolution_rate", 0)),
        "pr_merge_rate": float(m.get("pr_merge_rate", 0)),
        "commit_frequency": float(m.get("commit_frequency", 0)),
        "stalled_issues": int(m.get("stalled_issues", 0)),
        "unreviewed_prs": int(m.get("unreviewed_prs", 0)),
        "unique_authors": int(m.get("unique_authors", 0)),
    }

    return {
        "id": f"{owner}_{repo}_{sid}_summary",
        "content": "\n".join(content_parts),
        "metadata": metadata,
    }


def _build_issue_doc(sprint: dict, issue: dict, owner: str, repo: str) -> dict:
    sid = sprint["sprint_id"]
    num = issue.get("number", 0)
    labels = issue.get("labels", [])
    url = issue.get("url") or _github_url(owner, repo, "issue", num)

    content = (
        f"Issue #{num}: {issue.get('title', '')}\n"
        f"Repository: {owner}/{repo}\n"
        f"Sprint: {sid}\n"
        f"State: {issue.get('state', 'unknown')}\n"
        f"Labels: {', '.join(labels) if labels else 'none'}\n"
        f"URL: {url}\n\n"
        f"{(issue.get('body') or '')[:500]}"
    )

    return {
        "id": f"{owner}_{repo}_{sid}_issue_{num}",
        "content": content.strip(),
        "metadata": {
            "sprint_id": sid,
            "owner": owner,
            "repo": repo,
            "repo_full": f"{owner}/{repo}",
            "type": "issue",
            "number": num,
            "state": issue.get("state", ""),
            "labels": _safe_meta(labels),
            "url": url,
            "date": _safe_meta(issue.get("created_at", "")),
        },
    }


def _build_pr_doc(sprint: dict, pr: dict, owner: str, repo: str) -> dict:
    sid = sprint["sprint_id"]
    num = pr.get("number", 0)
    url = pr.get("url") or _github_url(owner, repo, "pr", num)

    content = (
        f"PR #{num}: {pr.get('title', '')}\n"
        f"Repository: {owner}/{repo}\n"
        f"Sprint: {sid}\n"
        f"State: {pr.get('state', 'unknown')}\n"
        f"Changes: +{pr.get('additions', 0)} -{pr.get('deletions', 0)}\n"
        f"URL: {url}\n\n"
        f"{(pr.get('body') or '')[:500]}"
    )

    return {
        "id": f"{owner}_{repo}_{sid}_pr_{num}",
        "content": content.strip(),
        "metadata": {
            "sprint_id": sid,
            "owner": owner,
            "repo": repo,
            "repo_full": f"{owner}/{repo}",
            "type": "pr",
            "number": num,
            "state": pr.get("state", ""),
            "additions": int(pr.get("additions", 0)),
            "deletions": int(pr.get("deletions", 0)),
            "url": url,
            "date": _safe_meta(pr.get("created_at", "")),
        },
    }


def _build_commit_doc(sprint: dict, commit: dict, owner: str, repo: str) -> dict:
    sid = sprint["sprint_id"]
    sha = commit.get("sha", "")
    sha_full = commit.get("sha_full", sha)
    url = commit.get("url") or _github_url(owner, repo, "commit", sha_full)

    raw_author = commit.get("author")
    if isinstance(raw_author, dict):
        author = raw_author.get("login") or raw_author.get("url") or "unknown"
    else:
        author = raw_author or "unknown"

    diff = commit.get("diff") or {}
    additions = diff.get("total_additions", 0)
    deletions = diff.get("total_deletions", 0)
    files_changed = diff.get("files_changed", 0)

    file_list = []
    for fd in (diff.get("file_diffs") or [])[:8]:
        file_list.append(
            f"  - {fd.get('filename', '')} [{fd.get('status', 'modified')}] "
            f"(+{fd.get('additions', 0)} -{fd.get('deletions', 0)})"
        )

    content = (
        f"Commit {sha} by {author}\n"
        f"Repository: {owner}/{repo}\n"
        f"Sprint: {sid}\n"
        f"URL: {url}\n\n"
        f"{commit.get('message', '')}\n\n"
        f"Code changes: +{additions} -{deletions} across {files_changed} files\n"
        f"Files:\n" + ("\n".join(file_list) if file_list else "  - none")
    )

    return {
        "id": f"{owner}_{repo}_{sid}_commit_{sha}",
        "content": content.strip(),
        "metadata": {
            "sprint_id": sid,
            "owner": owner,
            "repo": repo,
            "repo_full": f"{owner}/{repo}",
            "type": "commit",
            "sha": sha,
            "sha_full": sha_full,
            "author": author,
            "additions": int(additions),
            "deletions": int(deletions),
            "files_changed": int(files_changed),
            "url": url,
            "date": _safe_meta(commit.get("created_at", "")),
        },
    }


def _build_synthetic_summary_doc(sprint: dict) -> dict:
    """Build document for a synthetic sprint (no commits/issues/PRs detail)."""
    sid = sprint["sprint_id"]
    m = sprint.get("metrics", {})
    risk = sprint.get("risk_label", {})
    repo_slug = sprint.get("repo", "synthetic/repo")
    owner, repo = _extract_owner_repo(repo_slug)

    content_parts = [
        f"Synthetic sprint {sid} ({sprint.get('persona', 'unknown')} pattern)",
        f"Repository: {repo_slug}",
        "",
        "Metrics:",
        f"  Duration: {m.get('days_span', 0)} days",
        f"  Issues: {m.get('total_issues', 0)} total, {m.get('closed_issues', 0)} closed",
        f"  PRs: {m.get('total_prs', 0)} total, {m.get('merged_prs', 0)} merged",
        f"  Commits: {m.get('total_commits', 0)}",
        f"  Resolution rate: {m.get('issue_resolution_rate', 0):.2%}",
        f"  Merge rate: {m.get('pr_merge_rate', 0):.2%}",
        f"  Code changes: +{m.get('total_additions', 0)} -{m.get('total_deletions', 0)}",
        f"  Stalled issues: {m.get('stalled_issues', 0)}, Unreviewed PRs: {m.get('unreviewed_prs', 0)}",
        f"  Unique authors: {m.get('unique_authors', 0)}, participation: {m.get('author_participation', 0):.2f}",
        "",
        f"Risk score: {risk.get('risk_score', 0):.2f}, at-risk: {risk.get('is_at_risk', False)}",
    ]
    if risk.get("risk_factors"):
        content_parts.append(f"Risk factors: {', '.join(str(r) for r in risk['risk_factors'] if r)}")

    return {
        "id": f"synthetic_{sid}_summary",
        "content": "\n".join(content_parts),
        "metadata": {
            "sprint_id": sid,
            "owner": owner,
            "repo": repo,
            "repo_full": repo_slug,
            "type": "sprint_summary",
            "is_synthetic": True,
            "persona": _safe_meta(sprint.get("persona", "")),
            "risk_score": float(risk.get("risk_score", 0)),
            "is_at_risk": bool(risk.get("is_at_risk", False)),
            "total_issues": int(m.get("total_issues", 0)),
            "total_prs": int(m.get("total_prs", 0)),
            "total_commits": int(m.get("total_commits", 0)),
            "issue_resolution_rate": float(m.get("issue_resolution_rate", 0)),
            "pr_merge_rate": float(m.get("pr_merge_rate", 0)),
            "stalled_issues": int(m.get("stalled_issues", 0)),
            "unique_authors": int(m.get("unique_authors", 0)),
        },
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SprintChromaDB:
    """
    Full ChromaDB pipeline for sprint intelligence RAG.

    Handles:
    - Ingestion of real sprint files (with issues, PRs, commits)
    - Ingestion of synthetic sprint files (metrics only)
    - Similarity search by sprint context (text-based, using ChromaDB's
      built-in all-MiniLM-L6-v2 embeddings)
    - Evidence retrieval with GitHub citation URLs
    """

    def __init__(self, db_path: str | None = None):
        db_path = db_path or str(CHROMA_DIR)
        self._fix_onnx_cache()
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_SPRINTS,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"SprintChromaDB ready – collection '{COLLECTION_SPRINTS}' "
            f"has {self.collection.count()} documents"
        )

    # ------------------------------------------------------------------
    # ONNX cache fix (ChromaDB default embedding model)
    # ------------------------------------------------------------------
    @staticmethod
    def _fix_onnx_cache():
        import shutil
        cache_path = os.path.expanduser("~/.cache/chroma/onnx_models/all-MiniLM-L6-v2")
        if os.path.exists(cache_path) and not os.path.isdir(cache_path):
            os.remove(cache_path)
        elif os.path.isdir(cache_path) and not os.listdir(cache_path):
            shutil.rmtree(cache_path)

    # ==================================================================
    # Ingestion
    # ==================================================================

    def ingest_sprint_file(self, file_path: str | Path) -> int:
        """
        Load a sprint JSON file and upsert all documents into ChromaDB.

        Supports both real sprint files (with issues/prs/commits) and
        synthetic sprint files (metrics only).

        Returns number of documents upserted.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return 0

        logger.info(f"📂 Loading sprint file: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            sprints = json.load(f)

        if not isinstance(sprints, list):
            logger.warning(f"Expected list of sprints in {file_path}")
            return 0

        logger.info(f"📊 Found {len(sprints)} sprints in {file_path.name}")
        total = 0
        for idx, sprint in enumerate(sprints, 1):
            docs_count = self._ingest_single_sprint(sprint)
            total += docs_count
            if idx % max(1, len(sprints) // 10) == 0 or idx == 1:  # Log every 10% progress
                logger.info(f"  ⏳ Progress: {idx}/{len(sprints)} sprints processed ({total} documents)")

        logger.info(f"✅ Ingested {total} documents from {file_path.name}")
        return total

    def _ingest_single_sprint(self, sprint: dict) -> int:
        """Ingest one sprint record (real or synthetic) into ChromaDB."""
        repo_slug = sprint.get("repo", "synthetic/repo")
        owner, repo = _extract_owner_repo(repo_slug)
        sprint_id = sprint.get("sprint_id", "unknown")
        is_synthetic = "synthetic" in sprint_id or repo_slug == "synthetic/repo"

        docs: list[dict] = []

        if is_synthetic:
            logger.debug(f"  📈 Building synthetic sprint: {sprint_id}")
            docs.append(_build_synthetic_summary_doc(sprint))
        else:
            logger.debug(f"  🔍 Processing sprint: {sprint_id} ({owner}/{repo})")
            docs.append(_build_sprint_summary_doc(sprint, owner, repo))

            issues_count = len(sprint.get("issues", []))
            for issue in sprint.get("issues", []):
                docs.append(_build_issue_doc(sprint, issue, owner, repo))

            prs_count = len(sprint.get("prs", []))
            for pr in sprint.get("prs", []):
                docs.append(_build_pr_doc(sprint, pr, owner, repo))

            commits_count = len(sprint.get("commits", []))
            for commit in sprint.get("commits", []):
                docs.append(_build_commit_doc(sprint, commit, owner, repo))

            logger.debug(f"    📝 Documents: 1 summary + {issues_count} issues + {prs_count} PRs + {commits_count} commits = {len(docs)} docs")

        self._batch_upsert(docs, sprint_id=sprint_id)
        return len(docs)

    def _batch_upsert(self, docs: list[dict], batch_size: int = 200, sprint_id: str = ""):
        """Upsert documents in batches to stay within ChromaDB limits."""
        total_batches = (len(docs) + batch_size - 1) // batch_size
        logger.debug(f"    🚀 Upserting {len(docs)} documents in {total_batches} batch(es) to ChromaDB")

        for batch_num, i in enumerate(range(0, len(docs), batch_size), 1):
            batch = docs[i : i + batch_size]
            ids = [d["id"] for d in batch]
            contents = [d["content"] for d in batch]
            metadatas = [d["metadata"] for d in batch]

            self.collection.upsert(ids=ids, documents=contents, metadatas=metadatas)
            logger.debug(f"      ✓ Batch {batch_num}/{total_batches}: {len(batch)} documents upserted")

    def ingest_all_data(self, data_dir: str | Path | None = None) -> dict[str, int]:
        """
        Bulk-ingest all sprint JSON files from the data directory.

        Processes:
        - *_sprints.json  (real sprint data with issues/PRs/commits)
        - synthetic_sprints*.json (synthetic scenarios)

        Returns dict mapping filename → document count.
        """
        data_dir = Path(data_dir) if data_dir else DATA_DIR
        results: dict[str, int] = {}

        sprint_files = sorted(data_dir.glob("*_sprints*.json"))
        if not sprint_files:
            logger.warning(f"No sprint files found in {data_dir}")
            return results

        logger.info(f"🚀 Starting RAG ingestion pipeline...")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"📋 Found {len(sprint_files)} sprint files to process")
        logger.info(f"   Files: {', '.join(fp.name for fp in sprint_files)}")

        for file_num, fp in enumerate(sprint_files, 1):
            logger.info(f"\n[{file_num}/{len(sprint_files)}] Processing {fp.name}...")
            count = self.ingest_sprint_file(fp)
            results[fp.name] = count
            logger.info(f"   ✅ {fp.name}: {count} documents")

        total = sum(results.values())
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 RAG INGESTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"✓ Total documents ingested: {total}")
        logger.info(f"✓ Files processed: {len(results)}")
        for filename, count in results.items():
            logger.info(f"  - {filename}: {count} documents")
        logger.info(f"✓ Collection: {COLLECTION_SPRINTS}")
        logger.info(f"✓ Ready for RAG queries and agent inference")
        logger.info(f"{'='*60}\n")
        return results

    def get_collection_stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        count = self.collection.count()
        return {"collection": COLLECTION_SPRINTS, "total_documents": count}

    def purge_all(self) -> dict[str, Any]:
        """
        Purge all documents from the ChromaDB collection.

        WARNING: This is destructive and cannot be undone. All documents will be deleted.

        Returns:
            {"purged": True, "collection": collection_name, "documents_deleted": count}
        """
        try:
            # Get all document IDs
            results = self.collection.get(limit=100000)
            if results and results.get("ids"):
                all_ids = results["ids"]

                # Delete in batches
                batch_size = 1000
                total_deleted = 0
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    self.collection.delete(ids=batch_ids)
                    total_deleted += len(batch_ids)
                    logger.info(f"  🗑️ Deleted {total_deleted}/{len(all_ids)} documents...")

                logger.warning(f"🔥 ChromaDB purged: {total_deleted} documents deleted from '{COLLECTION_SPRINTS}'")
                return {
                    "purged": True,
                    "collection": COLLECTION_SPRINTS,
                    "documents_deleted": total_deleted,
                }
            else:
                logger.info(f"ℹ️  Collection '{COLLECTION_SPRINTS}' is already empty")
                return {
                    "purged": True,
                    "collection": COLLECTION_SPRINTS,
                    "documents_deleted": 0,
                }
        except Exception as e:
            logger.error(f"❌ Purge failed: {e}")
            return {
                "purged": False,
                "error": str(e),
            }

    # ==================================================================
    # RAG Retrieval
    # ==================================================================

    def query_similar_sprints(
        self,
        owner: str,
        repo: str,
        sprint_id: str | None = None,
        features: dict | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Find similar historical sprints for RAG context injection.

        Builds a rich query from owner/repo/sprint context + features.
        Returns full documents and metadata—not just IDs—so the LLM
        receives real historical context with citation URLs.

        Returns:
            {
                "similar_sprints": [{sprint_id, similarity, content, metadata, url}, ...],
                "evidence_citations": ["https://github.com/...", ...],
                "context_text": "formatted text block for LLM prompt injection",
            }
        """
        query_text = self._build_query_text(owner, repo, sprint_id, features)

        repo_full = f"{owner}/{repo}" if owner and repo else ""
        where_filter: dict[str, Any]
        if repo_full:
            # Keep retrieval repo-scoped to prevent cross-repo citation leakage.
            where_filter = {
                "$and": [
                    {"type": "sprint_summary"},
                    {"repo_full": repo_full},
                ]
            }
        else:
            where_filter = {"type": "sprint_summary"}

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(k * 3, 30),  # over-fetch, then filter
                where=where_filter,
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return {"similar_sprints": [], "evidence_citations": [], "context_text": ""}

        if not results or not results.get("ids") or not results["ids"][0]:
            return {"similar_sprints": [], "evidence_citations": [], "context_text": ""}

        similar: list[dict] = []
        seen_sprints: set[str] = set()

        for doc_id, distance, metadata, content in zip(
            results["ids"][0],
            results.get("distances", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("documents", [[]])[0],
        ):
            sid = metadata.get("sprint_id", "")
            # Skip self
            if sprint_id and sid == sprint_id and metadata.get("repo_full") == f"{owner}/{repo}":
                continue
            if sid in seen_sprints:
                continue
            seen_sprints.add(sid)

            similarity = max(0.0, 1.0 - distance) if distance is not None else 0.0
            similar.append({
                "sprint_id": sid,
                "similarity": round(similarity, 4),
                "repo": metadata.get("repo_full", ""),
                "content": content,
                "metadata": metadata,
                "is_synthetic": metadata.get("is_synthetic", False),
                "risk_score": metadata.get("risk_score", 0),
                "is_at_risk": metadata.get("is_at_risk", False),
            })

            if len(similar) >= k:
                break

        # Gather evidence citations from the matched sprints' related artifacts
        citations = self._gather_citations(similar[:3])

        context_text = self._format_rag_context(similar, citations)

        return {
            "similar_sprints": similar,
            "evidence_citations": citations,
            "context_text": context_text,
        }

    def get_sprint_evidence(
        self,
        owner: str,
        repo: str,
        sprint_id: str,
        risk_factors: list[str] | None = None,
        k: int = 10,
    ) -> dict[str, Any]:
        """
        Retrieve specific issues/PRs/commits from a sprint for evidence citation.

        Used by the ExplainerAgent to ground explanations in real artifacts.

        Returns:
            {
                "issues": [{number, title, state, url}, ...],
                "prs": [{number, title, state, url}, ...],
                "commits": [{sha, message, author, url}, ...],
            }
        """
        repo_full = f"{owner}/{repo}"

        evidence: dict[str, list] = {"issues": [], "prs": [], "commits": []}

        # Query for artifacts in this sprint
        try:
            where_filter: dict[str, Any] = {
                "$and": [
                    {"repo_full": repo_full},
                    {"sprint_id": sprint_id},
                ]
            }

            results = self.collection.get(
                where=where_filter,
                limit=100,
            )
        except Exception as e:
            logger.warning(f"Evidence fetch failed: {e}")
            return evidence

        if not results or not results.get("ids"):
            return evidence

        for metadata, content in zip(
            results.get("metadatas", []),
            results.get("documents", []),
        ):
            doc_type = metadata.get("type", "")
            url = metadata.get("url", _github_url(owner, repo, doc_type, metadata.get("number") or metadata.get("sha", "")))

            if doc_type == "issue":
                evidence["issues"].append({
                    "number": metadata.get("number"),
                    "state": metadata.get("state", ""),
                    "url": url,
                    "content_preview": content[:200] if content else "",
                })
            elif doc_type == "pr":
                evidence["prs"].append({
                    "number": metadata.get("number"),
                    "state": metadata.get("state", ""),
                    "additions": metadata.get("additions", 0),
                    "deletions": metadata.get("deletions", 0),
                    "url": url,
                    "content_preview": content[:200] if content else "",
                })
            elif doc_type == "commit":
                evidence["commits"].append({
                    "sha": metadata.get("sha", ""),
                    "author": metadata.get("author", ""),
                    "url": url,
                    "content_preview": content[:200] if content else "",
                })

        return evidence

    def query_by_risk(
        self,
        risk_keywords: str,
        owner: str | None = None,
        repo: str | None = None,
        k: int = 5,
    ) -> list[dict]:
        """
        Search for sprints matching specific risk patterns.
        Useful for finding precedent interventions.
        """
        query = f"Sprint risk: {risk_keywords}"

        where_filter = None
        if owner and repo:
            where_filter = {"repo_full": f"{owner}/{repo}"}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter,
                where_document={"$contains": risk_keywords.split()[0]} if risk_keywords.strip() else None,
            )
        except Exception as e:
            logger.warning(f"Risk query failed: {e}")
            return []

        matches = []
        if results and results.get("ids") and results["ids"][0]:
            for doc_id, distance, metadata, content in zip(
                results["ids"][0],
                results.get("distances", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("documents", [[]])[0],
            ):
                similarity = max(0.0, 1.0 - distance) if distance is not None else 0.0
                matches.append({
                    "id": doc_id,
                    "similarity": round(similarity, 4),
                    "content": content[:500],
                    "metadata": metadata,
                })
        return matches

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_query_text(
        self,
        owner: str,
        repo: str,
        sprint_id: str | None,
        features: dict | None,
    ) -> str:
        """Build semantic query text from sprint context + features."""
        parts = [f"Sprint analysis for {owner}/{repo}"]
        if sprint_id:
            parts.append(f"Sprint: {sprint_id}")

        if features:
            temporal = features.get("temporal", {})
            activity = features.get("activity", {})
            code = features.get("code", {})
            risk = features.get("risk", {})
            team = features.get("team", {})

            if activity:
                parts.append(
                    f"Issues: {activity.get('total_issues', 0):.0f}, "
                    f"PRs: {activity.get('total_prs', 0):.0f}, "
                    f"Commits: {activity.get('total_commits', 0):.0f}"
                )
                parts.append(
                    f"Resolution rate: {activity.get('issue_resolution_rate', 0):.2%}, "
                    f"Merge rate: {activity.get('pr_merge_rate', 0):.2%}"
                )
            if risk:
                risk_parts = []
                if risk.get("stalled_issues", 0) > 0:
                    risk_parts.append(f"stalled issues: {risk['stalled_issues']:.0f}")
                if risk.get("unreviewed_prs", 0) > 0:
                    risk_parts.append(f"unreviewed PRs: {risk['unreviewed_prs']:.0f}")
                if risk.get("abandoned_prs", 0) > 0:
                    risk_parts.append(f"abandoned PRs: {risk['abandoned_prs']:.0f}")
                if risk_parts:
                    parts.append(f"Risk signals: {', '.join(risk_parts)}")
            if team:
                parts.append(f"Team: {team.get('unique_authors', 0):.0f} authors")

        return "\n".join(parts)

    def _gather_citations(self, matches: list[dict[str, Any]]) -> list[str]:
        """Collect GitHub URLs from related artifacts of matched sprint+repo pairs."""
        citations: list[str] = []
        if not matches:
            return citations

        for match in matches:
            sid = str(match.get("sprint_id", "") or "")
            repo_full = str(
                (match.get("metadata") or {}).get("repo_full")
                or match.get("repo")
                or ""
            )
            if not sid or not repo_full:
                continue

            owner, repo = _extract_owner_repo(repo_full)

            try:
                results = self.collection.get(
                    where={
                        "$and": [
                            {"sprint_id": sid},
                            {"repo_full": repo_full},
                            {"type": {"$ne": "sprint_summary"}},
                        ]
                    },
                    limit=12,
                )
                if results and results.get("metadatas"):
                    for meta in results["metadatas"]:
                        url = str(meta.get("url", "") or "")
                        if not url:
                            doc_type = str(meta.get("type", "") or "")
                            number_or_sha = meta.get("number") or meta.get("sha") or meta.get("sha_full")
                            if owner and repo and doc_type and number_or_sha:
                                url = _github_url(owner, repo, doc_type, number_or_sha)
                        if url and url.startswith("https://") and url not in citations:
                            citations.append(url)
            except Exception:
                continue

        return citations[:15]  # Cap at 15 citations

    def _format_rag_context(
        self, similar: list[dict], citations: list[str]
    ) -> str:
        """Format retrieved context into a text block for LLM prompt injection."""
        if not similar:
            return "No similar historical sprints found."

        parts = ["## Similar Historical Sprints\n"]

        for i, s in enumerate(similar, 1):
            parts.append(f"### Case {i}: {s['sprint_id']} (similarity: {s['similarity']:.2f})")
            parts.append(f"Repository: {s['repo']}")
            parts.append(f"Risk score: {s['risk_score']:.2f}, At-risk: {s['is_at_risk']}")
            if s.get("is_synthetic"):
                parts.append("(Synthetic scenario)")
            # Include the actual sprint summary content
            if s.get("content"):
                parts.append(s["content"])
            parts.append("")

        if citations:
            parts.append("### Evidence URLs")
            for url in citations[:10]:
                parts.append(f"- {url}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point – run `python -m src.chromadb` to ingest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest sprint data into ChromaDB for RAG"
    )
    parser.add_argument(
        "--org",
        help="Organization name (e.g., Mintplex-Labs). If set, ingest only this org's repos."
    )
    parser.add_argument(
        "--repo",
        help="Repository name (e.g., anything-llm). Requires --org to be set."
    )
    parser.add_argument(
        "--query-test",
        action="store_true",
        help="Run a test RAG query after ingestion"
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="⚠️  DESTRUCTIVE: Delete ALL documents from ChromaDB collection"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)-8s | %(message)s"
    )

    # Handle purge command
    if args.purge:
        print("\n" + "="*70)
        print("⚠️  CHROMADB PURGE - DESTRUCTIVE OPERATION")
        print("="*70 + "\n")

        db = SprintChromaDB()
        stats_before = db.get_collection_stats()
        print(f"⚠️  About to delete ALL {stats_before['total_documents']} documents")
        print(f"   From collection: {stats_before['collection']}\n")

        # Confirmation
        confirm = input("Type 'PURGE' to confirm (or anything else to cancel): ").strip()
        if confirm != "PURGE":
            print("❌ Purge cancelled.\n")
            exit(0)

        result = db.purge_all()
        print(f"\n📊 Purge result:")
        print(f"   Documents deleted: {result.get('documents_deleted', 0)}")
        print(f"   Collection: {result.get('collection', 'unknown')}")
        print(f"   Status: {'✅ Success' if result.get('purged') else '❌ Failed'}\n")
        print("="*70 + "\n")
        exit(0)

    print("\n" + "="*70)
    print("🚀 SPRINT INTELLIGENCE RAG INGESTION PIPELINE")
    print("="*70 + "\n")

    db = SprintChromaDB()
    stats_before = db.get_collection_stats()
    print(f"📊 Database status BEFORE ingestion:")
    print(f"   Collection: {stats_before['collection']}")
    print(f"   Total documents: {stats_before['total_documents']}\n")

    # Ingest specific repo or all data
    if args.org and args.repo:
        # Ingest specific org/repo sprint files
        org_repo = f"{args.org}_{args.repo}"
        file_path = DATA_DIR / f"{org_repo}_sprints.json"

        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            print(f"   Expected file: {org_repo}_sprints.json")
            exit(1)

        print(f"🎯 Ingestion mode: SPECIFIC REPOSITORY")
        print(f"   Organization: {args.org}")
        print(f"   Repository: {args.repo}")
        print(f"   File: {file_path.name}\n")

        count = db.ingest_sprint_file(file_path)
    else:
        # Ingest all data
        print(f"🎯 Ingestion mode: ALL REPOSITORIES\n")
        results = db.ingest_all_data()

    stats_after = db.get_collection_stats()

    print(f"\n📊 Database status AFTER ingestion:")
    print(f"   Collection: {stats_after['collection']}")
    print(f"   Total documents: {stats_after['total_documents']}")
    docs_added = stats_after['total_documents'] - stats_before['total_documents']
    print(f"   Documents added: {docs_added}\n")

    # Optional test query
    if args.query_test and args.org and args.repo:
        print(f"{'='*70}")
        print("🧪 RUNNING TEST RAG QUERY")
        print(f"{'='*70}\n")

        logger.info(f"📍 Querying: {args.org}/{args.repo} sprint_005")
        rag = db.query_similar_sprints(
            owner=args.org, repo=args.repo, sprint_id="sprint_005"
        )
        print(f"✅ Test query completed successfully!")
        print(f"   Similar sprints found: {len(rag['similar_sprints'])}")
        print(f"   Evidence citations: {len(rag['evidence_citations'])}")
        if rag["similar_sprints"]:
            top = rag['similar_sprints'][0]
            print(f"   Top match: {top['sprint_id']} (similarity: {top['similarity']:.3f})")
        print()

    print("="*70)
    print("✨ RAG INGESTION PIPELINE COMPLETE")
    print("="*70 + "\n")
