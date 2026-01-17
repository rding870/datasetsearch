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
from huggingface_hub import HfApi, list_datasets, snapshot_download
from PIL import Image
import random
from dotenv import load_dotenv

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
    download_status: str
    multimodal_summary: Optional[str]
    sample_analysis: Optional[List[Dict[str, Any]]]


class HFSearchTool:
    """Tool for discovering datasets on Hugging Face Hub"""

    def __init__(self, hf_token: Optional[str] = None):
        self.api = HfApi(token=hf_token)

    def search_datasets(self, keywords: List[str], max_results: int = 50, apply_filter: bool = False) -> List[Dict[str, Any]]:
        """Search for image datasets matching criteria

        Args:
            keywords: List of keywords to search for
            max_results: Maximum number of results to return
            apply_filter: If True, only search image-classification datasets
        """
        print(f"🔍 Searching for datasets with keywords: {keywords}")
        if apply_filter:
            print(f"🔧 Filter enabled: image-classification only")

        results = []

        # Try searching with keyword-based queries first
        for keyword in keywords:
            try:
                # Build search parameters
                search_params = {
                    "search": keyword,
                    "sort": "downloads",
                    "direction": -1,
                    "limit": max_results
                }

                # Add filter if requested
                if apply_filter:
                    search_params["filter"] = "task_categories:image-classification"

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

            metadata = {
                'id': dataset_id,
                'description': getattr(dataset_info, 'description', ''),
                'tags': dataset_info.tags or [],
                'card_data': dataset_info.card_data.__dict__ if dataset_info.card_data else {},
                'readme': readme[:2000],  # Truncate for brevity
                'siblings': [f.rfilename for f in (dataset_info.siblings or [])[:10]],
                'downloads': getattr(dataset_info, 'downloads', 0),
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

    def calculate_heuristic_score(self, keywords: List[str], metadata: Dict[str, Any], require_english: bool = True) -> float:
        """Simple heuristic scoring with keyword relevance"""
        score = 0.0

        # Keyword relevance check (MOST IMPORTANT)
        dataset_text = f"{metadata.get('id', '')} {metadata.get('description', '')} {metadata.get('readme', '')}".lower()
        tags_text = ' '.join(metadata.get('tags', [])).lower()

        # Debug: Print what we're searching
        print(f"🔍 DEBUG - Checking keywords: {keywords}")
        print(f"🔍 DEBUG - Dataset ID in text: '{metadata.get('id', '')}'")
        print(f"🔍 DEBUG - Description length: {len(metadata.get('description', ''))}")
        print(f"🔍 DEBUG - README length: {len(metadata.get('readme', ''))}")

        keyword_matches = sum(1 for kw in keywords if kw.lower() in dataset_text or kw.lower() in tags_text)
        print(f"🔍 DEBUG - Keyword matches found: {keyword_matches}")

        # If no keywords match, return 0 (not relevant)
        if keyword_matches == 0:
            return 0.0

        # Check for English language (optional filter)
        if require_english:
            # Check if dataset has English language tag
            tags = metadata.get('tags', [])
            has_english_tag = any('english' in str(tag).lower() or 'en' == str(tag).lower() for tag in tags)

            # Check card_data for language info
            card_data = metadata.get('card_data', {})
            languages = card_data.get('language', []) if isinstance(card_data.get('language'), list) else [card_data.get('language', '')]
            has_english_in_card = any('en' in str(lang).lower() or 'english' in str(lang).lower() for lang in languages if lang)

            # If dataset explicitly specifies non-English only, penalize heavily
            if tags and not has_english_tag and not has_english_in_card:
                # Check for other language tags
                non_english_languages = ['korean', 'ko', 'chinese', 'zh', 'japanese', 'ja', 'french', 'fr', 'german', 'de']
                has_non_english = any(lang in str(tags).lower() for lang in non_english_languages)
                if has_non_english:
                    score *= 0.3  # Reduce score significantly for non-English datasets

        # Strong keyword match bonus
        keyword_ratio = keyword_matches / len(keywords)
        score += keyword_ratio * 0.5  # Up to 0.5 points for keyword matches

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

        prompt = f"""Evaluate this Hugging Face image dataset for relevance to the specified keywords.

**REQUIRED KEYWORDS:** {', '.join(keywords)}

Dataset ID: {metadata['id']}
Tags: {metadata.get('tags', [])}
Downloads: {metadata.get('downloads', 0)}
Description: {metadata.get('description', 'N/A')[:500]}
README excerpt: {metadata.get('readme', 'N/A')[:1000]}

CRITICAL REQUIREMENTS:
1. The dataset MUST be directly related to at least one of the keywords: {', '.join(keywords)}
2. If the keywords include specific subjects (e.g., "potato", "flower", "car"), the dataset MUST contain images of those subjects
3. Generic image classification datasets are NOT acceptable unless they specifically cover the keywords
4. The dataset documentation SHOULD be primarily in English (prefer datasets with English READMEs and descriptions)

Provide:
1. Does this dataset SPECIFICALLY contain or focus on the required keywords? (Yes/No)
2. Which keywords (if any) does this dataset cover?
3. Is the documentation primarily in English?
4. Brief reasoning (2-3 sentences)

Format: DECISION: [Yes/No] | KEYWORDS_FOUND: [list] | LANGUAGE: [English/Other] | REASONING: [your reasoning]"""

        try:
            message = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text
            is_useful = 'DECISION: Yes' in response

            print(f"✅ LLM Decision: {'Useful' if is_useful else 'Not useful'}")
            return is_useful, response

        except Exception as e:
            print(f"❌ LLM evaluation failed: {e}")
            return False, f"Evaluation failed: {e}"


class DownloaderTool:
    """Tool for downloading datasets"""

    def __init__(self, hf_token: Optional[str] = None, download_dir: str = "./datasets"):
        self.hf_token = hf_token
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def download_dataset(self, dataset_id: str, sample_only: bool = True) -> str:
        """Download dataset (or sample)"""
        print(f"⬇️  Downloading dataset: {dataset_id}")

        try:
            if sample_only:
                # Only download a subset for analysis
                path = snapshot_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    token=self.hf_token,
                    local_dir=os.path.join(self.download_dir, dataset_id.replace('/', '_')),
                    allow_patterns=["*.jpg", "*.png", "*.jpeg"],
                    max_workers=2,
                )
            else:
                path = snapshot_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    token=self.hf_token,
                    local_dir=os.path.join(self.download_dir, dataset_id.replace('/', '_')),
                )

            print(f"✅ Downloaded to: {path}")
            return f"Downloaded: {path}"

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return f"Failed: {e}"


class MultimodalAnalyzer:
    """Multimodal LLM analyzer for image understanding"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def analyze_images(self, dataset_path: str, num_samples: int = 3) -> Dict[str, Any]:
        """Analyze sample images from dataset"""
        print(f"🖼️  Analyzing images from: {dataset_path}")

        # Find image files
        image_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
                if len(image_files) >= num_samples:
                    break
            if len(image_files) >= num_samples:
                break

        if not image_files:
            print("❌ No images found")
            return {'summary': 'No images found', 'samples': []}

        # Sample random images
        sampled = random.sample(image_files, min(num_samples, len(image_files)))

        # Analyze images (metadata only for minimal version)
        samples = []
        for img_path in sampled:
            try:
                with Image.open(img_path) as img:
                    samples.append({
                        'path': img_path,
                        'size': img.size,
                        'format': img.format,
                        'mode': img.mode,
                    })
            except Exception as e:
                samples.append({'path': img_path, 'error': str(e)})

        summary = f"Analyzed {len(samples)} images. Formats: {set(s.get('format', 'unknown') for s in samples)}"
        print(f"✅ {summary}")

        return {
            'summary': summary,
            'samples': samples
        }


class AgentOrchestrator:
    """Central orchestrator for the agentic pipeline"""

    def __init__(self, hf_token: Optional[str] = None, anthropic_api_key: Optional[str] = None):
        self.search_tool = HFSearchTool(hf_token)
        self.reader_tool = DatasetReaderTool(hf_token)
        self.evaluator = EvaluatorTool(anthropic_api_key) if anthropic_api_key else None
        self.downloader = DownloaderTool(hf_token)
        self.analyzer = MultimodalAnalyzer(anthropic_api_key) if anthropic_api_key else None
        self.records: List[DatasetRecord] = []

    def run_pipeline(self, keywords: List[str], max_datasets: int = 10, download_useful: bool = True, apply_filter: bool = False):
        """Execute the full pipeline

        Args:
            keywords: Keywords to search for
            max_datasets: Maximum number of datasets to process
            download_useful: Whether to download datasets marked as useful
            apply_filter: If True, only search image-classification datasets
        """
        print("\n" + "="*60)
        print("🚀 Starting Agentic Dataset Discovery Pipeline")
        print("="*60 + "\n")

        # Step 1: Search
        datasets = self.search_tool.search_datasets(keywords, max_results=max_datasets, apply_filter=apply_filter)

        for dataset_info in datasets[:max_datasets]:
            dataset_id = dataset_info['id']
            print(f"\n{'─'*60}")
            print(f"Processing: {dataset_id}")
            print(f"{'─'*60}")

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

            # Step 4: Download if useful
            download_status = "skipped"
            multimodal_summary = None
            sample_analysis = None

            if is_useful and download_useful:
                download_status = self.downloader.download_dataset(dataset_id, sample_only=True)

                # Step 5: Analyze images
                if self.analyzer and "Downloaded:" in download_status:
                    dataset_path = download_status.split("Downloaded: ")[1]
                    analysis = self.analyzer.analyze_images(dataset_path)
                    multimodal_summary = analysis['summary']
                    sample_analysis = analysis['samples']

            # Create record
            record = DatasetRecord(
                dataset_id=dataset_id,
                discovered_at=datetime.now().isoformat(),
                metadata=metadata,
                heuristic_score=heuristic_score,
                llm_evaluation=llm_evaluation,
                is_useful=is_useful,
                download_status=download_status,
                multimodal_summary=multimodal_summary,
                sample_analysis=sample_analysis
            )

            self.records.append(record)
            print(f"✅ Record created for {dataset_id}")

        # Save results
        self.save_results()
        self.generate_report()

    def save_results(self, output_file: str = "dataset_records.json"):
        """Save structured records to JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
        print(f"\n💾 Saved {len(self.records)} records to {output_file}")

    def generate_report(self, output_file: str = "discovery_report.md"):
        """Generate research artifact report"""
        useful_count = sum(1 for r in self.records if r.is_useful)

        # Handle division by zero
        success_rate = (useful_count / len(self.records) * 100) if len(self.records) > 0 else 0.0

        report = f"""# Hugging Face Image Dataset Discovery Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Datasets Discovered:** {len(self.records)}
- **Datasets Marked Useful:** {useful_count}
- **Success Rate:** {success_rate:.1f}%

## Datasets Analyzed

"""
        for record in self.records:
            status = "✅ Useful" if record.is_useful else "❌ Not Useful"
            report += f"\n### {record.dataset_id} {status}\n"
            report += f"- **Heuristic Score:** {record.heuristic_score:.2f}\n"
            report += f"- **Downloads:** {record.metadata.get('downloads', 0)}\n"
            report += f"- **Download Status:** {record.download_status}\n"

            if record.llm_evaluation:
                report += f"\n**LLM Evaluation:**\n```\n{record.llm_evaluation[:300]}\n```\n"

            if record.multimodal_summary:
                report += f"\n**Multimodal Analysis:** {record.multimodal_summary}\n"

        report += "\n## Observations\n\n"
        report += "- Dataset documentation quality varies significantly\n"
        report += "- Automated evaluation helps filter high-quality datasets\n"
        report += "- Multimodal analysis provides valuable insights into dataset composition\n"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 Generated report: {output_file}")


class Scheduler:
    """Scheduler for running the pipeline periodically"""

    def __init__(self, orchestrator: AgentOrchestrator, config: Dict[str, Any]):
        """
        Args:
            orchestrator: The AgentOrchestrator instance to run
            config: Configuration dict with keys:
                - keywords: List[str]
                - max_datasets: int
                - download_useful: bool
                - apply_filter: bool
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
                download_useful=self.config.get('download_useful', True),
                apply_filter=self.config.get('apply_filter', False)
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
            print(f"\n\n{'='*60}")
            print(f"🛑 Scheduler stopped by user")
            print(f"📊 Total runs completed: {self.run_count}")
            print(f"{'='*60}\n")


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
            'keywords': ["dog"],
            'max_datasets': 5,
            'download_useful': True,
            'apply_filter': False,
            'interval_hours': 6  # Run every 6 hours (4 times per day)
        }

        scheduler = Scheduler(orchestrator, scheduler_config)
        scheduler.start(run_immediately=True)
    else:
        # One-time run mode
        orchestrator.run_pipeline(
            keywords=["dog"],
            max_datasets=5,
            download_useful=True,
            apply_filter=False
        )

        print("\n" + "="*60)
        print("✅ Pipeline completed successfully!")
        print("="*60)


if __name__ == "__main__":
    main()
