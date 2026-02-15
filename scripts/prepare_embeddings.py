#!/usr/bin/env python3
"""
Prepare and purify GitHub data for ChromaDB embeddings.

This script:
1. Extracts relevant text from GitHub events
2. Cleans and normalizes text
3. Generates embeddings using sentence-transformers
4. Stores in ChromaDB for RAG retrieval

Usage:
    python prepare_embeddings.py --input-dir ../data/raw/events --output-dir ../data/processed
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Text cleaning patterns
URL_PATTERN = re.compile(r'https?://\S+')
MENTION_PATTERN = re.compile(r'@[\w-]+')
CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```')
INLINE_CODE_PATTERN = re.compile(r'`[^`]+`')
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
EMOJI_PATTERN = re.compile(r':[a-z_]+:')
MULTIPLE_SPACES = re.compile(r'\s+')
SPECIAL_CHARS = re.compile(r'[^\w\s.,!?-]')


class TextCleaner:
    """Clean and normalize text for embeddings."""

    @staticmethod
    def clean_text(text: str, preserve_code: bool = False) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text to clean
            preserve_code: If False, removes code blocks and inline code

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = HTML_TAG_PATTERN.sub(' ', text)

        # Handle code blocks
        if not preserve_code:
            text = CODE_BLOCK_PATTERN.sub(' [CODE] ', text)
            text = INLINE_CODE_PATTERN.sub(' [CODE] ', text)

        # Replace URLs with placeholder
        text = URL_PATTERN.sub(' [URL] ', text)

        # Replace mentions with placeholder (preserve context)
        text = MENTION_PATTERN.sub(' [USER] ', text)

        # Replace emojis with placeholder
        text = EMOJI_PATTERN.sub(' ', text)

        # Remove special characters (keep punctuation)
        text = SPECIAL_CHARS.sub(' ', text)

        # Normalize whitespace
        text = MULTIPLE_SPACES.sub(' ', text)

        # Trim and lowercase
        text = text.strip().lower()

        return text

    @staticmethod
    def truncate_text(text: str, max_length: int = 512) -> str:
        """Truncate text to max length (for embedding models)."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


class GitHubEventExtractor:
    """Extract and prepare text from GitHub events."""

    def __init__(self, cleaner: TextCleaner):
        self.cleaner = cleaner

    def extract_from_issue(self, event: Dict) -> Optional[Dict]:
        """Extract text from IssuesEvent."""
        payload = event.get('payload', {})
        issue = payload.get('issue', {})

        if not issue:
            return None

        title = issue.get('title', '')
        body = issue.get('body', '')
        labels = [label.get('name', '') for label in issue.get('labels', [])]

        # Combine text
        text = f"{title}. {body}"
        cleaned_text = self.cleaner.clean_text(text)

        if not cleaned_text or len(cleaned_text) < 10:
            return None

        metadata = {
            'repo': event.get('repo', {}).get('name', ''),
            'issue_number': str(issue.get('number', '')),
            'state': issue.get('state', ''),
            'created_at': issue.get('created_at', ''),
            'event_created_at': event.get('created_at', ''),
            'author': event.get('actor', {}).get('login', ''),
        }
        # Only add labels if non-empty (ChromaDB requirement)
        if labels:
            metadata['labels'] = ','.join(labels)  # Convert to string

        return {
            'id': f"issue_{issue.get('id')}",
            'type': 'issue',
            'text': self.cleaner.truncate_text(cleaned_text),
            'metadata': metadata
        }

    def extract_from_pr(self, event: Dict) -> Optional[Dict]:
        """Extract text from PullRequestEvent."""
        payload = event.get('payload', {})
        pr = payload.get('pull_request', {})

        if not pr:
            return None

        title = pr.get('title', '')
        body = pr.get('body', '')

        # Combine text
        text = f"{title}. {body}"
        cleaned_text = self.cleaner.clean_text(text)

        if not cleaned_text or len(cleaned_text) < 10:
            return None

        return {
            'id': f"pr_{pr.get('id')}",
            'type': 'pull_request',
            'text': self.cleaner.truncate_text(cleaned_text),
            'metadata': {
                'repo': event.get('repo', {}).get('name', ''),
                'pr_number': str(pr.get('number', '')),
                'state': pr.get('state', ''),
                'action': payload.get('action', ''),
                'merged': str(pr.get('merged', False)),
                'created_at': pr.get('created_at', ''),
                'event_created_at': event.get('created_at', ''),
                'author': event.get('actor', {}).get('login', ''),
            }
        }

    def extract_from_comment(self, event: Dict) -> Optional[Dict]:
        """Extract text from IssueCommentEvent."""
        payload = event.get('payload', {})
        comment = payload.get('comment', {})
        issue = payload.get('issue', {})

        if not comment:
            return None

        body = comment.get('body', '')
        cleaned_text = self.cleaner.clean_text(body)

        if not cleaned_text or len(cleaned_text) < 10:
            return None

        metadata = {
            'repo': event.get('repo', {}).get('name', ''),
            'created_at': comment.get('created_at', ''),
            'event_created_at': event.get('created_at', ''),
            'author': event.get('actor', {}).get('login', ''),
        }
        if issue and issue.get('number'):
            metadata['issue_number'] = str(issue.get('number'))

        return {
            'id': f"comment_{comment.get('id')}",
            'type': 'comment',
            'text': self.cleaner.truncate_text(cleaned_text),
            'metadata': metadata
        }

    def extract_from_pr_review(self, event: Dict) -> Optional[Dict]:
        """Extract text from PullRequestReviewEvent."""
        payload = event.get('payload', {})
        review = payload.get('review', {})
        pr = payload.get('pull_request', {})

        if not review:
            return None

        body = review.get('body', '')
        cleaned_text = self.cleaner.clean_text(body)

        metadata = {
            'repo': event.get('repo', {}).get('name', ''),
            'state': review.get('state', ''),
            'created_at': review.get('submitted_at', ''),
            'event_created_at': event.get('created_at', ''),
            'author': event.get('actor', {}).get('login', ''),
        }
        if pr and pr.get('number'):
            metadata['pr_number'] = str(pr.get('number'))

        return {
            'id': f"review_{review.get('id')}",
            'type': 'pr_review',
            'text': self.cleaner.truncate_text(cleaned_text),
            'metadata': metadata
        }

    def extract_from_event(self, event: Dict) -> Optional[Dict]:
        """Extract text from any GitHub event."""
        event_type = event.get('type')

        extractors = {
            'IssuesEvent': self.extract_from_issue,
            'PullRequestEvent': self.extract_from_pr,
            'IssueCommentEvent': self.extract_from_comment,
            'PullRequestReviewEvent': self.extract_from_pr_review,
            'PullRequestReviewCommentEvent': self.extract_from_comment,
        }

        extractor = extractors.get(event_type)
        if extractor:
            return extractor(event)

        return None


class ChromaDBPreparer:
    """Prepare and store embeddings in ChromaDB."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 chroma_dir: str = '../data/processed/chromadb',
                 collection_name: str = 'github_events'):
        """
        Initialize ChromaDB preparer.

        Args:
            model_name: Sentence transformer model name (384-dim)
            chroma_dir: Directory to store ChromaDB data
            collection_name: Collection name in ChromaDB
        """
        self.model_name = model_name
        print(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        # Initialize ChromaDB
        chroma_path = Path(chroma_dir)
        chroma_path.mkdir(parents=True, exist_ok=True)

        print(f"🗄️  Initializing ChromaDB at: {chroma_path.absolute()}")
        self.client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        print(f"   Collection: {collection_name} ({self.collection.count()} existing items)")

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def add_to_chromadb(self, documents: List[Dict], batch_size: int = 100):
        """
        Add documents to ChromaDB.

        Args:
            documents: List of document dicts with 'id', 'text', 'metadata'
            batch_size: Batch size for adding to ChromaDB
        """
        if not documents:
            print("⚠️  No documents to add")
            return

        print(f"\n📊 Processing {len(documents)} documents...")

        # Process in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding to ChromaDB"):
            batch = documents[i:i + batch_size]

            ids = [doc['id'] for doc in batch]
            texts = [doc['text'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]

            # Generate embeddings
            embeddings = self.generate_embeddings(texts)

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

        print(f"✅ Added {len(documents)} documents to ChromaDB")
        print(f"   Total items in collection: {self.collection.count()}")

    def query_similar(self, query: str, n_results: int = 5) -> Dict:
        """Query for similar documents."""
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results


def process_event_files(input_dir: Path,
                       output_dir: Path,
                       max_files: Optional[int] = None,
                       max_docs_per_file: Optional[int] = None) -> List[Dict]:
    """
    Process GitHub event files and extract text documents.

    Args:
        input_dir: Directory containing JSONL event files
        output_dir: Directory to save processed data
        max_files: Maximum number of files to process
        max_docs_per_file: Maximum documents to extract per file

    Returns:
        List of document dictionaries
    """
    cleaner = TextCleaner()
    extractor = GitHubEventExtractor(cleaner)

    event_files = sorted(input_dir.glob('*.jsonl'))
    if max_files:
        event_files = event_files[:max_files]

    print(f"📂 Found {len(event_files)} event files to process")

    all_documents = []
    seen_ids = set()

    for event_file in tqdm(event_files, desc="Processing files"):
        docs_from_file = 0

        with open(event_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if max_docs_per_file and docs_from_file >= max_docs_per_file:
                    break

                try:
                    event = json.loads(line)
                    doc = extractor.extract_from_event(event)

                    if doc and doc['id'] not in seen_ids:
                        all_documents.append(doc)
                        seen_ids.add(doc['id'])
                        docs_from_file += 1

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    if line_num % 10000 == 0:
                        print(f"⚠️  Error at line {line_num}: {e}")
                    continue

    print(f"\n✅ Extracted {len(all_documents)} unique documents")

    # Save processed documents
    output_file = output_dir / 'extracted_documents.jsonl'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for doc in all_documents:
            f.write(json.dumps(doc) + '\n')

    print(f"💾 Saved to: {output_file}")

    # Print statistics
    type_counts = {}
    for doc in all_documents:
        doc_type = doc['type']
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

    print("\n📊 Document Type Distribution:")
    for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {doc_type}: {count}")

    return all_documents


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GitHub data for ChromaDB embeddings"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw/events',
        help='Directory containing raw GitHub event files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Directory to save processed data'
    )
    parser.add_argument(
        '--chroma-dir',
        type=str,
        default='data/processed/chromadb',
        help='Directory for ChromaDB storage'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model name'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of event files to process'
    )
    parser.add_argument(
        '--max-docs-per-file',
        type=int,
        default=1000,
        help='Maximum documents to extract per file'
    )
    parser.add_argument(
        '--skip-embedding',
        action='store_true',
        help='Skip embedding generation (only extract and clean)'
    )

    args = parser.parse_args()

    # Adjust paths if running from scripts directory
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        if Path('scripts').exists():
            input_dir = Path('.') / args.input_dir
        else:
            input_dir = Path('..') / args.input_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        if Path('scripts').exists():
            output_dir = Path('.') / args.output_dir
        else:
            output_dir = Path('..') / args.output_dir

    print("=" * 70)
    print("GitHub Data Preparation for ChromaDB Embeddings")
    print("=" * 70)
    print()

    # Step 1: Extract and clean documents
    documents = process_event_files(
        input_dir=input_dir,
        output_dir=output_dir,
        max_files=args.max_files,
        max_docs_per_file=args.max_docs_per_file
    )

    if not documents:
        print("❌ No documents extracted. Exiting.")
        return

    # Step 2: Generate embeddings and store in ChromaDB
    if not args.skip_embedding:
        print("\n" + "=" * 70)
        print("Generating Embeddings and Storing in ChromaDB")
        print("=" * 70)
        print()

        preparer = ChromaDBPreparer(
            model_name=args.model,
            chroma_dir=args.chroma_dir
        )

        preparer.add_to_chromadb(documents)

        # Test query
        print("\n" + "=" * 70)
        print("Testing Similarity Search")
        print("=" * 70)

        test_query = "authentication bug login error"
        print(f"\n🔍 Query: '{test_query}'")
        results = preparer.query_similar(test_query, n_results=3)

        print("\n📋 Top 3 Similar Documents:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"\n{i}. Similarity: {1 - distance:.3f}")
            print(f"   Repo: {metadata.get('repo', 'N/A')}")
            print(f"   Type: {metadata.get('type', 'N/A')}")
            print(f"   Text: {doc[:150]}...")

    print("\n" + "=" * 70)
    print("✅ Processing Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
