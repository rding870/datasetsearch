"""
Run the pipeline as a scheduled service

This script runs the dataset discovery pipeline periodically at scheduled intervals.
Configure the schedule settings in main.py or override them here.
"""

from main import main

if __name__ == "__main__":
    # Run with scheduler enabled
    main(use_scheduler=True)
