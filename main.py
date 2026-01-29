"""
Agentic Pipeline for Automated Discovery and Understanding of Hugging Face Image Datasets
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import anthropic
from huggingface_hub import HfApi, list_datasets
from dotenv import load_dotenv
from database import DatasetDatabase
from scraper import scrape_dataset_page 

# Load environment variables from .env file
load_dotenv()
@dataclass
class DatasetRecord:
    """Structured record for each dataset"""
    dataset_id: str
    discovered_at: str
    metadata: Dict[str, Any]
    heuristic_score: float
    llm_evaluation: Optional[str]
    is_useful: bool

class HFSearchTool:
    """Tool for discovering datasets on Hugging Face Hub"""

    def __init__(self, hf_token: Optional[str] = None):
        self.api = HfApi(token=hf_token)

    def search_datasets(self, max_per_keyword: int = 500) -> List[Dict[str, Any]]:
        """Search for image datasets matching criteria

        Args:
            keywords: List of keywords to search for
            max_per_keyword: Maximum results per keyword search (set high to get all)
        """
        results = []
        seen_ids = set()

        # Expand keywords with related search terms
        expanded_keywords = ["deepfake", "faceswap", "face forgery", "synthetic face", 
            "AI generated face", 'deepfake', 'deep-fake', 'deepfakes',
            'faceswap', 'face-swap', 'face swap',
            'face forgery', 'facial forgery',
            'face manipulation', 'face reenactment',
            'faceforensics', 'ff++',
            'dfdc', 'celeb-df', 'celebdf', 'deeperforensics', 'dfd',
            'fake face', 'fake faces',
            'face2face', 'neuraltextures',
            'deepfacelab', 'faceswap-gan',
            'ai generated', 'ai-generated',
            'synthetic face', 'synthetic faces',
            'gan face', 'gan generated',
            'stylegan', 'progan', 'diffusion face',
        ]

        # Remove duplicates while preserving order
        expanded_keywords = list(dict.fromkeys(expanded_keywords))

        print(f"Searching with {len(expanded_keywords)} keywords...")

        # Search each keyword
        for keyword in expanded_keywords:
            try:
                search_params = {
                    "search": keyword,
                    "sort": "downloads",
                    "direction": -1,
                    "limit": max_per_keyword,
                    "filter": "modality:image",
                }

                keyword_count = 0 # Shows how many new datasets each keyword contributed
                for dataset in list_datasets(**search_params):
                    dataset_id = dataset.id

                    if dataset_id in seen_ids:
                        continue

                    seen_ids.add(dataset_id)

                    dataset_info = {
                        'id': dataset_id,
                        'tags': dataset.tags or [],
                        'downloads': getattr(dataset, 'downloads', 0),
                        'likes': getattr(dataset, 'likes', 0),
                    }
                    results.append(dataset_info)
                    keyword_count += 1

                if keyword_count > 0:
                    print(f"  '{keyword}': +{keyword_count} new datasets")

            except Exception as e:
                print(f"  '{keyword}': failed - {e}")
                continue

        print(f"Found {len(results)} total unique datasets")
        return results

class DatasetReaderTool:
    """Tool for reading dataset documentation and metadata"""

    def __init__(self, hf_token: Optional[str] = None):
        self.api = HfApi(token=hf_token)

    def read_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Extract README, metadata, and structure"""
        print(f"📖 Reading dataset: {dataset_id}")

        try:
            dataset_info = self.api.dataset_info(dataset_id)

            # Get README content
            try:
                readme = self.api.get_dataset_readme(dataset_id)
            except Exception:
                readme = ""

            # Extract card_data
            card_data_dict = dataset_info.card_data.__dict__ if dataset_info.card_data else {}
            dataset_info_data = card_data_dict.get('dataset_info', {}) or {}
            if not isinstance(dataset_info_data, dict):
                dataset_info_data = {}

        

            # Separate structured tags (with ':') from user tags
            all_tags = dataset_info.tags or []
            user_tags = [tag for tag in all_tags if ':' not in tag]
            structured_tags = {tag.split(':', 1)[0]: tag.split(':', 1)[1] for tag in all_tags if ':' in tag}

            created_at = getattr(dataset_info, 'created_at', None)
            last_modified = getattr(dataset_info, 'last_modified', None)

            metadata = {
                'id': dataset_id,
                'description': getattr(dataset_info, 'description', ''),
                'tags': user_tags,
                'structured_tags': structured_tags,
                'readme': readme[:2000],
                'siblings': [f.rfilename for f in (dataset_info.siblings or [])[:10]],
                'downloads': getattr(dataset_info, 'downloads', 0),
                'likes': getattr(dataset_info, 'likes', 0),
                'created_at': created_at.isoformat() if created_at else None,
                'last_modified': last_modified.isoformat() if last_modified else None,
                'author': getattr(dataset_info, 'author', ''),
                'num_rows': None,
                'features': dataset_info_data.get('features', {}),
                'splits': None,
                'download_size': dataset_info_data.get('download_size'),
                'dataset_size': dataset_info_data.get('dataset_size'),
            }

            print(f"✅ Read metadata for {dataset_id}")
            return metadata

        except Exception as e:
            print(f"❌ Error reading dataset {dataset_id}: {e}")
            return {'id': dataset_id, 'error': str(e)}


class EvaluatorTool:
    """LLM-assisted evaluator for dataset usefulness"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def calculate_heuristic_score(self, metadata: Dict[str, Any]) -> float:
        """Quick scoring to prioritize datasets before LLM evaluation (0-1 scale)"""
        score = 0.0

        # Build searchable text from metadata
        dataset_text = f"{metadata.get('id', '')} {metadata.get('description', '')} {metadata.get('readme', '')}".lower()
        tags_text = ' '.join(metadata.get('tags', [])).lower()
        combined_text = dataset_text + ' ' + tags_text

        # All relevant keywords in one list
        keywords = [
            'deepfake', 'deep fake', 'deepfakes', 'faceswap', 'face swap', 'face-swap',
            'face forgery', 'facial forgery', 'face manipulation', 'video forgery',
            'dfdc', 'faceforensics', 'celeb-df', 'face2face', 'neuraltextures', 'deepfacelab',
            'synthetic', 'generated', 'gan', 'diffusion', 'stable diffusion', 'stylegan', 'progan',
            'dalle', 'dall-e', 'midjourney', 'ai-generated', 'ai generated', 'fake face', 'fake image',
            'detection', 'detector', 'forensic', 'forgery detection', 'authenticity',
            'biggan', 'vqgan', 'ldm', 'glide', 'flux'
        ]

        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in combined_text)
        if matches == 0:
            return 0.0

        # Keyword relevance (up to 0.5)
        score += min(matches / 5.0, 1.0) * 0.5

        # Popularity bonus (up to 0.2)
        downloads = metadata.get('downloads', 0)
        if downloads > 1000:
            score += 0.2
        elif downloads > 100:
            score += 0.1

        # README quality (up to 0.2)
        readme = metadata.get('readme', '')
        if readme and len(readme) > 100:
            score += 0.2

        # Image-related tags (up to 0.1)
        if any(tag in tags_text for tag in ['image-classification', 'computer-vision', 'image']):
            score += 0.1

        return min(score, 1.0)

    def llm_evaluate(self, metadata: Dict[str, Any]) -> tuple[bool, str]:
        """Use LLM to evaluate dataset usefulness"""
        print(f"🤖 LLM evaluating: {metadata['id']}")

        # Handle None values safely
        description = metadata.get('description') or 'N/A'
        readme = metadata.get('readme') or 'N/A'

        prompt = f"""Evaluate this dataset for training a computer vision ML system to distinguish between real imagery and deepfake/AI-generated imagery.

Dataset ID: {metadata['id']}
Author: {metadata.get('author', 'Unknown')}
Tags: {metadata.get('tags', [])}
Downloads: {metadata.get('downloads', 0)}
Likes: {metadata.get('likes', 0)}
Created: {metadata.get('created_at', 'Unknown')}
Last Modified: {metadata.get('last_modified', 'Unknown')}
Files: {metadata.get('siblings', [])[:5]}
Description: {description[:500]}
README excerpt: {readme[:2000]}

Evaluate the following criteria:

1. RELEVANCE: Is this dataset relevant for training deepfake/AI-gen detection?
   - Must contain BOTH real images AND deepfake/AI-generated images with labels (0=real, 1=fake)
   - Or contain only deepfakes/AI-gen if clearly labeled

2. QUALITY & DOCUMENTATION:
   - Is the dataset well-maintained with good documentation?
   - How/where was data collected? Is this documented?
   - Which deepfake methods or AI models were used to generate fake images? (FaceSwap, Stable Diffusion, etc.)

3. QUANTITY:
   - How many images total? (estimate from docs)
   - Image resolution (width x height)?

4. LABELS:
   - How are images labeled? (CSV file, folder structure, metadata, etc.)
   - Is labeling robust and clear? (0=real, 1=fake convention?)

5. POPULARITY & CREDIBILITY:
   - Download count: {metadata.get('downloads', 0)}
   - Connected to research papers?
   - Publication date/recency?

6. OTHER HELPFUL INFO:
   - Any concerns (data quality, bias, ethical issues)?
   - Unique strengths?

Format your response as:
DECISION: [Yes/No]
RELEVANCE: [High/Medium/Low] - [why]
QUALITY: [High/Medium/Low] - [documentation quality, generation methods used]
QUANTITY: [# images, resolution]
LABELS: [format, robustness]
CREDIBILITY: [downloads, papers, date]
CONCERNS: [any red flags]
STRENGTHS: [unique benefits]
REASONING: [2-3 sentence summary]"""

        try:
            message = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text
            # Check for Yes in the decision line (handles "Yes", "Tentative Yes", etc.)
            is_useful = 'DECISION:' in response and 'Yes' in response.split('DECISION:')[1].split('\n')[0]

            print(f"✅ LLM Decision: {'Useful' if is_useful else 'Not useful'}")
            print(f"📝 Reasoning: {response}")
            return is_useful, response

        except Exception as e:
            print(f"❌ LLM evaluation failed: {e}")
            return False, f"Evaluation failed: {e}"


class EmbeddingTool:
    """Tool for generating text embeddings using HuggingFace Inference API"""

    def __init__(self, hf_token: str, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=hf_token)
        self.model_id = model_id

    def create_embedding_text(self, dataset_id: str, description: str, readme: str, llm_evaluation: str) -> str:
        """Concatenate fields into a single text for embedding"""
        parts = [
            f"Dataset: {dataset_id}",
            f"Description: {description or 'N/A'}",
            f"README: {(readme or 'N/A')[:1000]}",
            f"Evaluation: {(llm_evaluation or 'N/A')[:500]}"
        ]
        return " | ".join(parts)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector from HuggingFace Inference API"""
        try:
            embedding = self.client.feature_extraction(text, model=self.model_id)
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            print(f"✅ Generated embedding (dim={len(embedding)})")
            return embedding
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            return None

    def embed_dataset(self, dataset_id: str, metadata: Dict[str, Any], llm_evaluation: Optional[str]) -> Optional[List[float]]:
        """Generate embedding for a dataset record"""
        text = self.create_embedding_text(
            dataset_id=dataset_id,
            description=metadata.get('description', ''),
            readme=metadata.get('readme', ''),
            llm_evaluation=llm_evaluation or ''
        )
        return self.get_embedding(text)


class DatasetAgent:
    """True AI agent that decides its own actions - same capabilities as pipeline"""

    def __init__(self):
        hf_token = os.getenv("HF_TOKEN")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.db = DatasetDatabase("datasets.db")
        self.search_tool = HFSearchTool(hf_token)
        self.reader_tool = DatasetReaderTool(hf_token)
        self.evaluator = EvaluatorTool(api_key)
        self.embedding_tool = EmbeddingTool(hf_token)
        self.pending_datasets = []

        self.tools = [
            {"name": "search_datasets", "description": "Search HuggingFace for deepfake/AI-generated image datasets",
             "input_schema": {"type": "object", "properties": {}, "required": []}},
            {"name": "evaluate_dataset", "description": "Full evaluation: read metadata, scrape, heuristic score, LLM evaluation, embedding, save to DB",
             "input_schema": {"type": "object", "properties": {"dataset_id": {"type": "string"}}, "required": []}},
            {"name": "export_data", "description": "Export database to CSV and Google Sheets",
             "input_schema": {"type": "object", "properties": {}, "required": []}},
            {"name": "finish", "description": "Stop the agent when task is complete",
             "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}
        ]

    def execute_tool(self, name: str, args: dict) -> str:
        if name == "search_datasets":
            results = self.search_tool.search_datasets(max_per_keyword=50)
            self.pending_datasets = [r['id'] for r in results]
            return f"Found {len(results)} datasets. Pending: {len(self.pending_datasets)}"

        elif name == "evaluate_dataset":
            dataset_id = args.get("dataset_id") or (self.pending_datasets.pop(0) if self.pending_datasets else None)
            if not dataset_id:
                return "No datasets to evaluate"
            # Same as pipeline: read → scrape → heuristic → LLM eval → embedding → save
            metadata = self.reader_tool.read_dataset(dataset_id)
            scraped = scrape_dataset_page(dataset_id)
            if 'error' not in scraped:
                metadata['full_readme'] = scraped.get('full_readme')
                metadata['scraped_data'] = scraped
            heuristic_score = self.evaluator.calculate_heuristic_score(metadata)
            is_useful, llm_evaluation = self.evaluator.llm_evaluate(metadata) if heuristic_score >= 5 else (False, None)
            record = DatasetRecord(
                dataset_id=dataset_id, discovered_at=datetime.now().isoformat(),
                metadata=metadata, heuristic_score=heuristic_score,
                llm_evaluation=llm_evaluation, is_useful=is_useful
            )
            embedding = self.embedding_tool.embed_dataset(dataset_id, metadata, llm_evaluation)
            self.db.insert_dataset(asdict(record), embedding=embedding)
            return f"Saved: {dataset_id} (score={heuristic_score}). Pending: {len(self.pending_datasets)}"

        elif name == "export_data":
            try:
                self.db.export_to_google_sheets("1I6xzTmSohoNaQPDd9yDnN8U5WIWRkc3fiS56NTL8BEk")
                return "Exported to Google Sheets"
            except Exception as e:
                return f"Google Sheets export failed: {e}"

        elif name == "finish":
            return "DONE"
        return "Unknown tool"

    def run(self, goal: str = "Find and evaluate deepfake detection datasets", max_steps: int = 500):
        print(f"🤖 Agent starting with goal: {goal}\n")
        messages = [{"role": "user", "content": f"Goal: {goal}. Use tools to search, evaluate datasets, check stats, or finish when done."}]

        for step in range(max_steps):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=1024,
                tools=self.tools, messages=messages
            )
            if response.stop_reason == "tool_use":
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"Step {step+1}: {block.name}({block.input})")
                        result = self.execute_tool(block.name, block.input)
                        print(f"  → {result}\n")
                        if result == "DONE":
                            print("✅ Agent finished")
                            return
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": block.id, "content": result}]})
                        # Keep messages under control: first message (goal) + last 30 messages
                        if len(messages) > 32:
                            messages = messages[:1] + messages[-30:]
            else:
                for block in response.content:
                    if hasattr(block, 'text'):
                        print(f"Agent: {block.text}")
                break
        print("⚠️ Max steps reached")


if __name__ == "__main__":
    agent = DatasetAgent()
    agent.run()
