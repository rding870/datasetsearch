"""
Agentic Pipeline for Automated Discovery and Understanding of Hugging Face Image Datasets
"""

import os
import json
import time
import schedule
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import anthropic
from huggingface_hub import HfApi, list_datasets
from PIL import Image
import random
from dotenv import load_dotenv
from database import DatasetDatabase

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

    def search_datasets(self, keywords: List[str], max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for image datasets matching criteria

        Args:
            keywords: List of keywords to search for
            max_results: Maximum number of results to return
        """
        results = []

        # Expand keywords with related search terms
        expanded_keywords = keywords.copy()

        # Add common dataset names if searching for deepfakes/synthetic media
        deepfake_dataset_names = ['faceforensics', 'dfdc', 'celeb-df', 'deepfakes',
                                   'dfdm', 'deeperforensics', 'genface', 'openfake']
        ai_gen_dataset_names = ['laion', 'dragon', 'synthetic', 'generated',
                               'diffusion', 'stable-diffusion', 'dalle']

        expanded_keywords.extend(deepfake_dataset_names)
        expanded_keywords.extend(ai_gen_dataset_names)

        # Try searching with keyword-based queries
        for keyword in expanded_keywords:
            try:
                # Build search parameters
                search_params = {
                    "search": keyword,
                    "sort": "downloads",
                    "direction": -1,
                    "limit": max_results
                }

                filters = [
                    "modality:image",
                ]
                    # Try each filter (HF API may only support one at a time)
                for filter_term in filters:
                    try:
                        search_params["filter"] = filter_term
                    except:
                        continue

                for dataset in list_datasets(**search_params):
                    # Basic heuristic filtering
                    dataset_info = {
                        'id': dataset.id,
                        'tags': dataset.tags or [],
                        'downloads': getattr(dataset, 'downloads', 0),
                        'likes': getattr(dataset, 'likes', 0),
                    }

                    # Avoid duplicates
                    if not any(r['id'] == dataset_info['id'] for r in results):
                        results.append(dataset_info)

                    if len(results) >= max_results:
                        break

            except Exception as e:
                print(f"⚠️  Search failed for keyword '{keyword}': {e}")
                continue

            if len(results) >= max_results:
                break

        print(f"✅ Found {len(results)} matching datasets")
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
            readme = ""
            try:
                readme = self.api.get_dataset_readme(dataset_id)
            except Exception:
                readme = "No README available"

            created_at = getattr(dataset_info, 'created_at', None)
            last_modified = getattr(dataset_info, 'last_modified', None)

            # Extract dataset_info and card_data
            num_rows = None
            features = {}
            splits = {}
            download_size = None
            dataset_size = None

            card_data_dict = dataset_info.card_data.__dict__ if dataset_info.card_data else {}

            # Extract structured info from dataset_info
            if 'dataset_info' in card_data_dict and card_data_dict['dataset_info']:
                dataset_info_data = card_data_dict['dataset_info']
                if isinstance(dataset_info_data, dict):
                    features = dataset_info_data.get('features', {})
                    splits_raw = dataset_info_data.get('splits', [])
                    download_size = dataset_info_data.get('download_size', None)
                    dataset_size = dataset_info_data.get('dataset_size', None)

                    # Calculate total rows from splits (can be dict or list)
                    if isinstance(splits_raw, dict):
                        splits = splits_raw
                        num_rows = sum(split.get('num_examples', 0) for split in splits.values() if isinstance(split, dict))
                        print(f"  Found splits (dict): {list(splits.keys())}, total rows: {num_rows}")
                    elif isinstance(splits_raw, list):
                        # Convert list to dict for storage: [{name: 'train', num_examples: 100}] -> {'train': {'num_examples': 100}}
                        splits = {s.get('name', f'split_{i}'): s for i, s in enumerate(splits_raw) if isinstance(s, dict)}
                        num_rows = sum(s.get('num_examples', 0) for s in splits_raw if isinstance(s, dict))
                        print(f"  Found splits (list): {[s.get('name', '?') for s in splits_raw]}, total rows: {num_rows}")
                        print(f"  Raw splits data: {splits_raw}")

            # Fallback: check alternative attributes
            if num_rows is None:
                # Debug: print what attributes are available
                print(f"  Debug: Available attributes for {dataset_id}:")
                print(f"    card_data keys: {list(card_data_dict.keys())}")
                if 'dataset_info' in card_data_dict:
                    print(f"    dataset_info type: {type(card_data_dict['dataset_info'])}")
                    if card_data_dict['dataset_info']:
                        print(f"    dataset_info content sample: {str(card_data_dict['dataset_info'])[:200]}")

                # Try common attributes
                for attr in ['num_rows', 'dataset_size', 'size']:
                    if hasattr(dataset_info, attr):
                        val = getattr(dataset_info, attr)
                        print(f"    Found {attr}: {val}")
                        if isinstance(val, int):
                            num_rows = val
                            break

            # Separate structured tags (with ':') from user tags
            all_tags = dataset_info.tags or []
            user_tags = [tag for tag in all_tags if ':' not in tag]
            structured_tags = {tag.split(':', 1)[0]: tag.split(':', 1)[1] for tag in all_tags if ':' in tag}

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
                'num_rows': num_rows,
                'features': features,
                'splits': splits,
                'download_size': download_size,
                'dataset_size': dataset_size,
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

    def calculate_heuristic_score(self, keywords: List[str], metadata: Dict[str, Any]) -> float:
        """Simple heuristic scoring with keyword relevance"""
        score = 0.0

        # Keyword relevance check 
        # Handle None values for text fields
        dataset_id = metadata.get('id', '') or ''
        description = metadata.get('description', '') or ''
        readme = metadata.get('readme', '') or ''

        dataset_text = f"{dataset_id} {description} {readme}".lower()
        tags_text = ' '.join(metadata.get('tags', [])).lower()

        # Expanded keyword matching with semantic relevance
        # Core terms for deepfake datasets
        deepfake_terms = ['deepfake', 'deep fake', 'faceswap', 'face swap', 'face-swap',
                          'face forgery', 'facial forgery', 'deepfakes', 'dfdc', 'faceforensics',
                          'celeb-df', 'face manipulation', 'video forgery']

        # Core terms for AI-generated imagery
        ai_gen_terms = ['synthetic', 'generated', 'gan', 'diffusion', 'stable diffusion',
                       'dalle', 'dall-e', 'midjourney', 'imagen', 'ai-generated',
                       'ai generated', 'text-to-image', 'text to image', 'aigc',
                       'fake face', 'fake image', 'generative', 'stylegan']

        # Detection-related terms
        detection_terms = ['detection', 'detector', 'forensic', 'forgery detection',
                          'fake detection', 'authenticity', 'manipulation detection']

        # Model names to look for
        model_names = ['faceswap', 'face2face', 'neuraltextures', 'deepfacelab',
                      'stylegan', 'progan', 'biggan', 'vqgan', 'ldm', 'glide', 'flux']

        # Combine all relevant terms
        all_relevant_terms = deepfake_terms + ai_gen_terms + detection_terms + model_names

        # Count keyword matches (original keywords)
        keyword_matches = sum(1 for kw in keywords if kw.lower() in dataset_text or kw.lower() in tags_text)

        # Count semantic matches (related terms)
        semantic_matches = sum(1 for term in all_relevant_terms if term.lower() in dataset_text or term.lower() in tags_text)

        # More flexible matching: accept if either keywords OR semantic terms match
        if keyword_matches == 0 and semantic_matches == 0:
            return 0.0

        # Strong keyword match bonus
        keyword_ratio = keyword_matches / len(keywords) if keywords else 0
        score += keyword_ratio * 0.5  # Up to 0.5 points for keyword matches

        # Semantic relevance bonus
        semantic_ratio = min(semantic_matches / 5.0, 1.0)  # Cap at 5 matches
        score += semantic_ratio * 0.3  # Up to 0.3 points for semantic matches

        # Downloads and likes
        downloads = metadata.get('downloads', 0)
        if downloads > 1000:
            score += 0.2
        elif downloads > 100:
            score += 0.1

        # Has README
        readme = metadata.get('readme', '')
        if readme and len(readme) > 100:
            score += 0.2

        # Image-related tags
        image_tags = ['image-classification', 'computer-vision', 'image']
        if any(tag in metadata.get('tags', []) for tag in image_tags):
            score += 0.1

        return min(score, 1.0)

    def llm_evaluate(self, keywords: List[str], metadata: Dict[str, Any]) -> tuple[bool, str]:
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

class AgentOrchestrator:
    """Central orchestrator for the agentic pipeline"""

    def __init__(self, hf_token: Optional[str] = None, anthropic_api_key: Optional[str] = None, db_path: str = "datasets.db"):
        self.search_tool = HFSearchTool(hf_token)
        self.reader_tool = DatasetReaderTool(hf_token)
        self.evaluator = EvaluatorTool(anthropic_api_key) if anthropic_api_key else None
        self.records: List[DatasetRecord] = []
        self.db = DatasetDatabase(db_path)

    def run_pipeline(self, keywords: List[str], max_datasets: int = 10):
        """Execute the full pipeline

        Args:
            keywords: Keywords to search for
            max_datasets: Maximum number of datasets to process
        """

        # Step 1: Search
        datasets = self.search_tool.search_datasets(keywords, max_results=max_datasets)

        for dataset_info in datasets[:max_datasets]:
            dataset_id = dataset_info['id']

            # Step 2: Read metadata
            metadata = self.reader_tool.read_dataset(dataset_id)

            if 'error' in metadata:
                continue

            # Step 3: Evaluate
            heuristic_score = self.evaluator.calculate_heuristic_score(keywords, metadata) if self.evaluator else 0.5
            print(f"📊 Heuristic Score: {heuristic_score:.2f}")

            is_useful = False
            llm_evaluation = None

            # Only proceed if heuristic score > 0 (keyword match required)
            if self.evaluator and heuristic_score > 0:
                is_useful, llm_evaluation = self.evaluator.llm_evaluate(keywords, metadata)
            else:
                print(f"⏭️  Skipped: No keyword match found")

            if is_useful:
                # Check if high quality - download and analyze samples
                image_quality_approved = True  # Default to true for datasets that don't require image check

                if llm_evaluation and 'RELEVANCE: High' in llm_evaluation and 'QUALITY: High' in llm_evaluation:
                    print(f"🌟 High-quality dataset detected! Downloading 20 sample images for analysis...")
                    try:
                        # Get list of files first without downloading
                        from huggingface_hub import HfFileSystem
                        fs = HfFileSystem()
                        files = fs.ls(f"datasets/{dataset_id}", detail=False, recursive=True)

                        # Filter for images
                        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                        # Sample 20 random images
                        if len(image_files) > 20:
                            sampled_files = random.sample(image_files, 20)
                        else:
                            sampled_files = image_files

                        # Download only the sampled files
                        download_dir = f"./datasets/{dataset_id.replace('/', '_')}"
                        os.makedirs(download_dir, exist_ok=True)

                        sampled = []
                        for file_path in sampled_files[:20]:
                            # Extract relative path
                            rel_path = file_path.replace(f"datasets/{dataset_id}/", "")
                            local_path = os.path.join(download_dir, os.path.basename(rel_path))

                            # Download single file
                            from huggingface_hub import hf_hub_download
                            hf_hub_download(
                                repo_id=dataset_id,
                                repo_type="dataset",
                                filename=rel_path,
                                local_dir=download_dir,
                                local_dir_use_symlinks=False
                            )
                            sampled.append(local_path)

                        if sampled:
                            print(f"🔍 Analyzing {len(sampled)} sample images...")

                            approved_count = 0
                            # Analyze all samples with Claude
                            for img_path in sampled:
                                try:
                                    with Image.open(img_path) as img:
                                        # Encode image for Claude
                                        import base64
                                        import io
                                        buffered = io.BytesIO()
                                        img.save(buffered, format="PNG")
                                        img_str = base64.b64encode(buffered.getvalue()).decode()

                                        # Ask Claude to analyze
                                        response = self.evaluator.client.messages.create(
                                            model="claude-3-5-haiku-20241022",
                                            max_tokens=300,
                                            messages=[{
                                                "role": "user",
                                                "content": [
                                                    {
                                                        "type": "image",
                                                        "source": {
                                                            "type": "base64",
                                                            "media_type": "image/png",
                                                            "data": img_str,
                                                        },
                                                    },
                                                    {
                                                        "type": "text",
                                                        "text": "Briefly assess image quality for ML training: resolution, clarity, any visible artifacts/issues? Answer with APPROVED or REJECTED followed by brief reason."
                                                    }
                                                ]
                                            }]
                                        )
                                        analysis = response.content[0].text
                                        if 'APPROVED' in analysis.upper():
                                            approved_count += 1
                                        print(f"  Sample {sampled.index(img_path)+1}: {analysis[:100]}...")
                                except Exception as e:
                                    print(f"  ⚠️ Could not analyze {img_path}: {e}")

                            approval_rate = approved_count / len(sampled) if sampled else 0
                            print(f"✅ Image quality check: {approved_count}/{len(sampled)} approved ({approval_rate:.1%})")

                            # Check if majority approved
                            if approval_rate < 0.5:
                                print(f"⏭️ Dataset rejected: Only {approval_rate:.1%} of images approved")
                                image_quality_approved = False
                    except Exception as e:
                        print(f"⚠️  Sample analysis failed: {e}")
                        image_quality_approved = False

                # Only add to database if both useful AND image quality approved
                if image_quality_approved:
                    # Create record
                    record = DatasetRecord(
                        dataset_id=dataset_id,
                        discovered_at=datetime.now().isoformat(),
                        metadata=metadata,
                        heuristic_score=heuristic_score,
                        llm_evaluation=llm_evaluation,
                        is_useful=is_useful,
                    )

                    self.records.append(record)
                    print(f"✅ Record created for {dataset_id}")

                    # Save to database
                    self.db.insert_dataset(asdict(record))

                    # Save results to JSON (legacy)
                    self.save_results()

    def save_results(self, output_file: str = "dataset_records.json"):
        """Save structured records to JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
        print(f"\n💾 Saved {len(self.records)} records to {output_file}")


class Scheduler:
    """Scheduler for running the pipeline periodically"""

    def __init__(self, orchestrator: AgentOrchestrator, config: Dict[str, Any]):
        """
        Args:
            orchestrator: The AgentOrchestrator instance to run
            config: Configuration dict with keys:
                - keywords: List[str]
                - max_datasets: int
                - interval_hours: int (default: 24)
        """
        self.orchestrator = orchestrator
        self.config = config
        self.run_count = 0

    def run_job(self):
        """Execute a single pipeline run"""
        self.run_count += 1
        print(f"\n{'='*60}")
        print(f"🕐 Scheduled Run #{self.run_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        try:
            self.orchestrator.run_pipeline(
                keywords=self.config.get('keywords', ['image']),
                max_datasets=self.config.get('max_datasets', 10),
            )
            print(f"\n✅ Scheduled run #{self.run_count} completed successfully")
        except Exception as e:
            print(f"\n❌ Scheduled run #{self.run_count} failed: {e}")

    def start(self, run_immediately: bool = True):
        """
        Start the scheduler

        Args:
            run_immediately: If True, run the pipeline once before starting the schedule
        """
        interval_hours = self.config.get('interval_hours', 24)

        print(f"\n{'='*60}")
        print(f"📅 Scheduler Started")
        print(f"{'='*60}")
        print(f"⏰ Interval: Every {interval_hours} hour(s)")
        print(f"🔑 Keywords: {self.config.get('keywords', [])}")
        print(f"📊 Max Datasets: {self.config.get('max_datasets', 10)}")
        print(f"{'='*60}\n")

        # Run immediately if requested
        if run_immediately:
            print("▶️  Running initial pipeline execution...")
            self.run_job()

        # Schedule periodic runs
        schedule.every(interval_hours).hours.do(self.run_job)

        print(f"\n⏳ Waiting for next scheduled run in {interval_hours} hour(s)...")
        print("Press Ctrl+C to stop the scheduler\n")

        # Keep the scheduler running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            pass


def main(use_scheduler: bool = False):
    """Main entry point

    Args:
        use_scheduler: If True, run as a scheduled service. If False, run once.
    """
    # Configuration
    HF_TOKEN = os.getenv("HF_TOKEN")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        hf_token=HF_TOKEN,
        anthropic_api_key=ANTHROPIC_API_KEY
    )

    if use_scheduler:
        # Scheduled mode - runs periodically
        scheduler_config = {
            'keywords': ["deepfake", "synthetic media", "face forgery", "AI generated"],
            'max_datasets': 20,
            'interval_hours': 6  # Run every 6 hours (4 times per day)
        }

        scheduler = Scheduler(orchestrator, scheduler_config)
        scheduler.start(run_immediately=True)
    else:
        # One-time run mode
        orchestrator.run_pipeline(
            keywords=["deepfake", "synthetic media", "face forgery", "AI generated"],
            max_datasets=20,
        )

if __name__ == "__main__":
    main()
