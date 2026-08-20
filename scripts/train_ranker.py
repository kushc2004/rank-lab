#!/usr/bin/env python3
from _stage_cli import run_stage


if __name__ == "__main__":
    run_stage("rankers", "Train natural, density-weighted, popularity-weighted, and relevance-ablation rankers.")
